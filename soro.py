import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure # 追加
import itertools

# ==========================================
# 設定と定数
# ==========================================
SUMOCFG_FILE = "4cross.sumocfg"  # 設定ファイル名
EPISODES = 10
SIMULATION_TIME = 5400
CYCLE_TIME = 200
ACTION_INTERVAL = 200  # 行動決定を行う間隔（秒）

# 信号機の設定
# 固定時間（変更不可なフェーズ）
FIXED_YELLOW_TIME = 3
FIXED_ALL_RED_TIME = 2
FIXED_LOST_PER_PHASE = FIXED_YELLOW_TIME + FIXED_ALL_RED_TIME  # 5秒
TOTAL_PHASES = 4
TOTAL_LOST_TIME = FIXED_LOST_PER_PHASE * TOTAL_PHASES  # 20秒

# 可変時間（学習対象）
ALLOCATABLE_TIME = CYCLE_TIME - TOTAL_LOST_TIME  # 180秒
TIME_UNIT = 15  # 1ブロックの時間
MIN_GREEN_TIME = 10 # 最小青時間（1ブロック15秒なので、1ブロック以上割り当てれば満たされる）
TOTAL_BLOCKS = ALLOCATABLE_TIME // TIME_UNIT  # 12ブロック

# 交差点IDリスト
TL_IDS = ["C1", "C2", "C3", "C4"]

# 各交差点の監視対象エッジ（流入路）
# 順序: 北, 西, 東, 南 (各3レーン想定)
# 4方向 * 3レーン = 12レーン
INCOMING_EDGES = {
    "C1": ["N1_C1", "W1_C1", "C2_C1", "C3_C1"],
    "C2": ["N2_C2", "C1_C2", "E1_C2", "C4_C2"],
    "C3": ["C1_C3", "W2_C3", "C4_C3", "S1_C3"],
    "C4": ["C2_C4", "C3_C4", "E2_C4", "S2_C4"],
}

# フェーズの定義 (State strings from user requirement)
# 0: 南北直進左折(G), 1: 黄, 2: 赤, 3: 南北右折(G), ...
# 修正: 黄色フェーズ(1, 7)での 'g' を 'y' に変更して、青->赤の直接遷移を防ぐ
PHASE_STATES = [
    "GGgrrrGGgrrr", # 0: Action0 (North-South Straight/Left)
    "yyyrrryyyrrr", # 1: Yellow (Fixed: g->y)
    "rrrrrrrrrrrr", # 2: All Red
    "rrGrrrrrGrrr", # 3: Action1 (North-South Right)
    "rryrrrrryrrr", # 4: Yellow
    "rrrrrrrrrrrr", # 5: All Red
    "rrrGGgrrrGGg", # 6: Action2 (East-West Straight/Left)
    "rrryyyrrryyy", # 7: Yellow (Fixed: g->y)
    "rrrrrrrrrrrr", # 8: All Red
    "rrrrrGrrrrrG", # 9: Action3 (East-West Right)
    "rrrrryrrrrry", # 10: Yellow
    "rrrrrrrrrrrr"  # 11: All Red
]

# ==========================================
# ヘルパー関数: 行動空間の生成
# ==========================================
def generate_valid_allocations():
    """
    12個のブロック(15秒/個)を4つのフェーズに分配する全パターンを生成する。
    制約: 各フェーズ最低1ブロック割り当てる (15秒 >= 最小10秒 を満たすため)
    """
    valid_actions = []
    # x0 + x1 + x2 + x3 = 12, xi >= 1
    # これは y0 + y1 + y2 + y3 = 8, yi >= 0 と同義 (xi = yi + 1)
    # 愚直にループで探索
    for p0 in range(1, TOTAL_BLOCKS - 2):
        for p1 in range(1, TOTAL_BLOCKS - 2):
            for p2 in range(1, TOTAL_BLOCKS - 2):
                for p3 in range(1, TOTAL_BLOCKS - 2):
                    if p0 + p1 + p2 + p3 == TOTAL_BLOCKS:
                        # 秒数に変換して保存
                        times = [p0*TIME_UNIT, p1*TIME_UNIT, p2*TIME_UNIT, p3*TIME_UNIT]
                        valid_actions.append(times)
    return valid_actions

VALID_ACTIONS = generate_valid_allocations()
NUM_ACTIONS = len(VALID_ACTIONS)
print(f"Action Space Size: {NUM_ACTIONS}") # Should be 165

# ==========================================
# 環境クラス定義
# ==========================================
class SumoIndependentEnv(gym.Env):
    """
    Gymnasium環境準拠だが、Stepメソッドは
    Dict[AgentID, Action]を受け取り、Dict[AgentID, Obs], ... を返す
    マルチエージェント仕様とする。
    """
    def __init__(self):
        super().__init__()
        
        # 観測空間: 各交差点 4方向x3レーンx2指標(台数, 時間) = 24次元
        self.observation_space = spaces.Box(low=0, high=99999, shape=(24,), dtype=np.float32)
        
        # 行動空間: 定義したパターンのインデックス
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        
        # SUMO設定
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
            
        self.sumo_cmd = [
            sumolib.checkBinary('sumo'), 
            '-c', SUMOCFG_FILE, 
            '--no-step-log', 'true', 
            '--waiting-time-memory', '1000',
            '--time-to-teleport', '-1' # テレポート無効化（渋滞を正しく計測するため）
        ]
        
        self.is_running = False
        self.episode_step = 0
        
        # 報酬計算用のメモリ
        self.prev_cycle_cost = {tl: 0.0 for tl in TL_IDS}

    def reset(self, seed=None, options=None):
        if self.is_running:
            traci.close()
        
        traci.start(self.sumo_cmd)
        self.is_running = True
        self.episode_step = 0
        
        # 初期化
        self.prev_cycle_cost = {tl: 0.0 for tl in TL_IDS}
        
        # 最初の観測を取得
        return self._get_observations()

    def step(self, actions_dict):
        """
        actions_dict: { "C1": action_idx, "C2": ... }
        200秒間シミュレーションを進め、報酬を計算する。
        """
        self.episode_step += 1
        
        # 1. 選択された行動から、各交差点のフェーズ時間を設定
        # VALID_ACTIONS[idx] -> [G0, G1, G2, G3] (秒数)
        current_green_times = {}
        for tl_id, action_idx in actions_dict.items():
            current_green_times[tl_id] = VALID_ACTIONS[action_idx]
        
        # 2. サイクル実行 (200秒)
        # 累積コスト（待ち台数の積分）を計算
        current_cycle_cost = {tl: 0.0 for tl in TL_IDS}
        
        # タイムステップごとのフェーズ制御ロジック
        # サイクル構造:
        # Phase 0 (G): t0秒 -> Phase 1 (Y): 3s -> Phase 2 (R): 2s
        # Phase 3 (G): t1秒 -> Phase 4 (Y): 3s -> Phase 5 (R): 2s
        # Phase 6 (G): t2秒 -> Phase 7 (Y): 3s -> Phase 8 (R): 2s
        # Phase 9 (G): t3秒 -> Phase 10 (Y): 3s -> Phase 11 (R): 2s
        
        # 各フェーズの終了時刻(サイクル内時刻)を計算
        phase_schedule = {tl: [] for tl in TL_IDS}
        for tl_id in TL_IDS:
            times = current_green_times[tl_id] # [t0, t1, t2, t3]
            # スケジュール作成 (Duration, PhaseIndex)
            schedule = []
            schedule.append((times[0], 0))
            schedule.append((FIXED_YELLOW_TIME, 1))
            schedule.append((FIXED_ALL_RED_TIME, 2))
            schedule.append((times[1], 3))
            schedule.append((FIXED_YELLOW_TIME, 4))
            schedule.append((FIXED_ALL_RED_TIME, 5))
            schedule.append((times[2], 6))
            schedule.append((FIXED_YELLOW_TIME, 7))
            schedule.append((FIXED_ALL_RED_TIME, 8))
            schedule.append((times[3], 9))
            schedule.append((FIXED_YELLOW_TIME, 10))
            schedule.append((FIXED_ALL_RED_TIME, 11))
            phase_schedule[tl_id] = schedule

        # 200秒ループ
        # 現在どのフェーズにいるか、そのフェーズの残り時間は何かを管理
        current_phase_idx = {tl: 0 for tl in TL_IDS}
        current_phase_time = {tl: 0 for tl in TL_IDS} # 経過時間
        
        # 初期フェーズセット
        for tl_id in TL_IDS:
            traci.trafficlight.setRedYellowGreenState(tl_id, PHASE_STATES[phase_schedule[tl_id][0][1]])

        for _ in range(CYCLE_TIME):
            # A. シミュレーション1ステップ進行
            traci.simulationStep()
            
            # B. 状態集計 (コスト計算)
            # コスト = 全流入レーンの停止台数の合計
            for tl_id in TL_IDS:
                halt_sum = 0
                for edge in INCOMING_EDGES[tl_id]:
                    for i in range(3):
                        lane_id = f"{edge}_{i}"
                        halt_sum += traci.lane.getLastStepHaltingNumber(lane_id)
                current_cycle_cost[tl_id] += halt_sum

            # C. 信号制御
            for tl_id in TL_IDS:
                current_phase_time[tl_id] += 1
                p_idx = current_phase_idx[tl_id]
                target_duration = phase_schedule[tl_id][p_idx][0]
                
                # フェーズ時間が終了したら次へ
                if current_phase_time[tl_id] >= target_duration:
                    current_phase_time[tl_id] = 0
                    current_phase_idx[tl_id] += 1
                    
                    if current_phase_idx[tl_id] < len(phase_schedule[tl_id]):
                        next_state_idx = phase_schedule[tl_id][current_phase_idx[tl_id]][1]
                        traci.trafficlight.setRedYellowGreenState(tl_id, PHASE_STATES[next_state_idx])

        # 3. 報酬計算と次状態取得
        rewards = {}
        next_obs = self._get_observations()
        
        for tl_id in TL_IDS:
            # 報酬 = 前回のコスト - 今回のコスト (良くなっていればプラス)
            # 初回は前回コストが0なので、大きなマイナスになる可能性があるが、学習が進めば相対評価になる
            # ただし初回ステップの挙動を安定させるため、prevが0の場合は報酬0とする等の処理も考えられるが
            # ここでは単純な差分とする。
            if self.episode_step == 1:
                reward = 0.0 # 初回は比較対象がないので0
            else:
                reward = self.prev_cycle_cost[tl_id] - current_cycle_cost[tl_id]
            
            rewards[tl_id] = reward
            self.prev_cycle_cost[tl_id] = current_cycle_cost[tl_id] # 更新

        # 終了判定
        done = traci.simulation.getTime() >= SIMULATION_TIME
        dones = {tl: done for tl in TL_IDS}
        infos = {tl: {"wait_cost": current_cycle_cost[tl]} for tl in TL_IDS}
        
        return next_obs, rewards, dones, infos
    
    def _get_observations(self):
        """全交差点の観測を取得"""
        observations = {}
        for tl_id in TL_IDS:
            obs_list = []
            for edge in INCOMING_EDGES[tl_id]:
                for i in range(3):
                    lane_id = f"{edge}_{i}"
                    try:
                        halt = traci.lane.getLastStepHaltingNumber(lane_id)
                        wait = traci.lane.getWaitingTime(lane_id)
                    except:
                        halt = 0
                        wait = 0
                    obs_list.extend([halt, wait])
            observations[tl_id] = np.array(obs_list, dtype=np.float32)
        return observations
        
    def close(self):
        traci.close()

# ==========================================
# メイン学習ループ
# ==========================================
if __name__ == "__main__":
    # 環境生成
    env = SumoIndependentEnv()
    
    # エージェント生成 (完全独立: 4つの異なるDQNモデル)
    # MLPポリシーを使用、verbose=0でログ抑制
    agents = {}
    for tl_id in TL_IDS:
        agents[tl_id] = DQN(
            "MlpPolicy", 
            env, # Dummy env passed for initialization
            learning_rate=1e-3, 
            buffer_size=10000, 
            learning_starts=100, 
            batch_size=32, 
            gamma=0.99, 
            target_update_interval=50,
            verbose=0
        )
        # 手動トレーニングループ用にロガーと状態を設定
        # これを行わないと train() 呼び出し時に AttributeError: 'DQN' object has no attribute '_logger' が発生する
        log_path = f"./logs/{tl_id}"
        new_logger = configure(log_path, ["stdout", "csv"])
        agents[tl_id].set_logger(new_logger)
        agents[tl_id]._current_progress_remaining = 1.0

    # 統計用
    all_episode_wait_times = {tl: [] for tl in TL_IDS}
    
    print("Start Training...")
    
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        total_rewards = {tl: 0 for tl in TL_IDS}
        total_wait_costs = {tl: 0 for tl in TL_IDS} # グラフ用
        steps = 0
        
        while not done:
            # 1. 行動選択
            actions = {}
            for tl_id in TL_IDS:
                # SB3のpredictは (action, state) を返す
                action, _ = agents[tl_id].predict(obs[tl_id], deterministic=False)
                actions[tl_id] = int(action)
            
            # 2. 環境進行
            next_obs, rewards, dones, infos = env.step(actions)
            
            # 3. 学習 (Experience Replay & Train)
            for tl_id in TL_IDS:
                # ReplayBufferに手動で追加
                # SB3のバッファ構造: add(obs, next_obs, action, reward, done, infos)
                agents[tl_id].replay_buffer.add(
                    obs[tl_id].reshape(1, -1),  # バッチ次元が必要な場合があるが、add内で処理されることが多い
                    next_obs[tl_id].reshape(1, -1), 
                    np.array([actions[tl_id]]), 
                    np.array([rewards[tl_id]]), 
                    np.array([dones[tl_id]]), 
                    [infos[tl_id]]
                )
                
                # 学習ステップを実行
                # train()メソッドは内部でgradient_steps回数の更新を行う
                # ただし、learning_startsを超えている必要がある
                agents[tl_id].train(gradient_steps=1, batch_size=32)
                
                # 統計更新
                total_rewards[tl_id] += rewards[tl_id]
                total_wait_costs[tl_id] += infos[tl_id]["wait_cost"]
            
            obs = next_obs
            done = all(dones.values())
            steps += 1
            
        # エピソード終了後の集計
        print(f"Episode {ep+1}/{EPISODES} Finished.")
        for tl_id in TL_IDS:
            # 平均待ちコスト (総コスト / サイクル数)
            avg_wait = total_wait_costs[tl_id] / steps
            all_episode_wait_times[tl_id].append(avg_wait)
            print(f"  {tl_id}: Total Reward={total_rewards[tl_id]:.2f}, Avg Cycle Cost={avg_wait:.2f}")

    env.close()
    
    # ==========================================
    # 結果のグラフ出力
    # ==========================================
    plt.figure(figsize=(10, 6))
    for tl_id in TL_IDS:
        plt.plot(range(1, EPISODES + 1), all_episode_wait_times[tl_id], label=tl_id, marker='o')
    
    plt.title("Average Waiting Cost per Cycle (Independent DQN)")
    plt.xlabel("Episode")
    plt.ylabel("Avg Waiting Cost (Halting Number Sum)")
    plt.legend()
    plt.grid(True)
    plt.savefig("waiting_time_result.png")
    print("Graph saved as waiting_time_result.png")