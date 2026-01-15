import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

# ==========================================
# 定数・設定定義
# ==========================================

# SUMOの設定
SUMO_CMD = ["sumo", "-c", "4cross.sumocfg", "--no-step-log", "true", "--waiting-time-memory", "1000"]
# GUIで確認したい場合は "sumo" を "sumo-gui" に変更してください

# シミュレーション設定
EPISODE_LENGTH = 5400  # 秒
CYCLE_TIME = 200       # 秒
CONTROL_STEP = 200     # 秒ごとに制御
TOTAL_EPISODES = 10    # テスト回数

# 交差点定義
JUNCTIONS = ["C1", "C2", "C3", "C4"]

# 隣接関係定義
NEIGHBORS = {
    "C1": ["C2", "C3"],
    "C2": ["C1", "C4"],
    "C3": ["C1", "C4"],
    "C4": ["C2", "C3"]
}

# 流入エッジ定義 (各交差点への流入路)
# 各エッジには3レーンあると仮定 (例: N1_C1_0, N1_C1_1, N1_C1_2)
INCOMING_EDGES = {
    "C1": ["N1_C1", "W1_C1", "C2_C1", "C3_C1"], # 北, 西, 東(C2), 南(C3)
    "C2": ["N2_C2", "E1_C2", "C1_C2", "C4_C2"], # 北, 東, 西(C1), 南(C4)
    "C3": ["S1_C3", "W2_C3", "C1_C3", "C4_C3"], # 南, 西, 北(C1), 東(C4)
    "C4": ["S2_C4", "E2_C4", "C2_C4", "C3_C4"]  # 南, 東, 北(C2), 西(C3)
}

# 行動パターンの定義
# 可変時間合計 180秒 (200 - 固定20) を4つのフェーズ(G0, G3, G6, G9)に分配
# 制約: G0, G6 >= 20, G3, G9 >= 0, 20秒刻み
# パターン形式: [Phase0(NS直), Phase3(NS右), Phase6(EW直), Phase9(EW右)]
ACTION_PATTERNS = [
    [70, 20, 70, 20],  # 0: バランス (NS=90, EW=90)
    [90, 20, 50, 20],  # 1: NSやや優先 (NS=110, EW=70)
    [50, 20, 90, 20],  # 2: EWやや優先 (NS=70, EW=110)
    [110, 20, 30, 20], # 3: NS強力優先 (NS=130, EW=50)
    [30, 20, 110, 20], # 4: EW強力優先 (NS=50, EW=130)
]
NUM_PATTERNS = len(ACTION_PATTERNS)

# フェーズ構成 (固定時間含む)
# Phase ID: 0(G), 1(Y), 2(R), 3(G), 4(Y), 5(R), 6(G), 7(Y), 8(R), 9(G), 10(Y), 11(R)
FIXED_PHASES = {
    1: 3, 2: 2,   # NS直左 黄/全赤
    4: 3, 5: 2,   # NS右   黄/全赤
    7: 3, 8: 2,   # EW直左 黄/全赤
    10: 3, 11: 2  # EW右   黄/全赤
}

# 報酬パラメータ
ALPHA = 0.5

# ==========================================
# Gymnasium 環境クラス
# ==========================================

class SumoTrafficEnv(gym.Env):
    def __init__(self):
        super(SumoTrafficEnv, self).__init__()

        # 行動空間: 4つの交差点 × 5パターン = 5^4 = 625通りの組み合わせ
        # Discrete空間として定義し、内部でデコードする
        self.action_space = spaces.Discrete(NUM_PATTERNS ** 4)

        # 観測空間: 4交差点 × 72次元 (各交差点の入力次元) = 288次元
        # 各交差点の入力: 自交差点(12レーン×2指標) + 隣接1(24) + 隣接2(24) = 72
        # 指標: 待ち台数, 待ち時間
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4 * 72,), dtype=np.float32
        )

        self.episode_step = 0
        self.total_waiting_time_history = {jc: [] for jc in JUNCTIONS}
        
        # 報酬計算用の前回値保持 {Junction: waiting_time}
        self.prev_waiting_time = {jc: 0.0 for jc in JUNCTIONS}

        # SUMOの接続確認
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # SUMOの起動またはリロード
        try:
            traci.close()
        except:
            pass
            
        traci.start(SUMO_CMD)
        
        self.episode_step = 0
        self.prev_waiting_time = {jc: 0.0 for jc in JUNCTIONS}
        self.total_waiting_time_history = {jc: [] for jc in JUNCTIONS}
        
        # 最初のウォームアップ（少し回して車両を発生させる）
        # 今回は0秒から制御開始するため即座に観測
        obs = self._get_observation()
        
        # 初回の待ち時間を記録
        for jc in JUNCTIONS:
            self.prev_waiting_time[jc] = self._get_junction_waiting_time(jc)

        return obs, {}

    def step(self, action_idx):
        # 1. 行動のデコード (Action Index -> 各交差点のパターンID)
        # 基数変換のようなロジックで分解
        patterns = []
        temp_action = action_idx
        for _ in range(4): # 4交差点
            patterns.append(temp_action % NUM_PATTERNS)
            temp_action //= NUM_PATTERNS
        patterns.reverse() # C1, C2, C3, C4 の順に

        junction_actions = dict(zip(JUNCTIONS, patterns))

        # 2. 信号サイクルの適用 (200秒間のシミュレーション)
        # SUMOではsetPhaseDurationは「現在のフェーズ」の長さを変えるか、
        # プログラム全体を書き換える必要がある。ここではフェーズごとにステップを進める簡易実装を行う。
        
        # 各交差点のフェーズ時間割を構築
        # key: jc, value: list of durations for phases 0..11
        durations = {}
        for i, jc in enumerate(JUNCTIONS):
            pat_idx = junction_actions[jc]
            g_times = ACTION_PATTERNS[pat_idx] # [G0, G3, G6, G9]
            
            d = [0] * 12
            d[0] = g_times[0]
            d[3] = g_times[1]
            d[6] = g_times[2]
            d[9] = g_times[3]
            
            for p_idx, fixed_dur in FIXED_PHASES.items():
                d[p_idx] = fixed_dur
            durations[jc] = d

        # 200秒（ステップ）分の進行
        # フェーズ順序: 0 -> 1 -> ... -> 11
        # 全交差点同期してフェーズ0から開始すると仮定し、traciで制御
        
        # 現在時刻から各フェーズの切り替わりタイミングを計算してsetPhaseDurationを発行する手もあるが、
        # 確実に同期させるため、1秒ずつ、あるいはフェーズごとに進める
        
        # 簡略化のため、サイクル開始時に全交差点をフェーズ0にリセット
        for jc in JUNCTIONS:
            traci.trafficlight.setPhase(jc, 0)
        
        current_phase_idx = 0
        # 12フェーズ分ループ
        for phase_id in range(12):
            # このフェーズの最大所要時間を探す（同期ズレを防ぐためWaitするならここで調整だが、今回は固定長サイクルなので全交差点合計時間は200で一致）
            # 各交差点ごとに duration をセット
            for jc in JUNCTIONS:
                duration = durations[jc][phase_id]
                traci.trafficlight.setPhase(jc, phase_id)
                traci.trafficlight.setPhaseDuration(jc, duration)
            
            # このフェーズの時間の最大値分進める（今回は設計上、全交差点で合計200になるが、フェーズごとの長さは違う）
            # ここでは簡単のため、SUMOの論理に任せて「全交差点の現在のフェーズ残り時間」を消費させるのではなく
            # 強制的に時間を刻む実装にする
            
            # 注意: setPhaseDurationは「現在のフェーズが終わるまでの時間」を設定する
            # 全交差点に対して設定した後、その時間分シミュレーションを進める必要がある
            # しかし、交差点によってフェーズ0の長さが違う（例: C1は70秒, C2は90秒）。
            # 違う長さの場合、短い交差点は次のフェーズへ移ってしまう。
            # これを厳密に制御するには毎秒 setPhase を呼ぶのが確実。
            
            pass 

        # --- 毎秒制御方式 ---
        # 0~199秒のタイムステップで、各秒ごとにどのフェーズにいるべきかを計算し適用
        for t in range(CYCLE_TIME):
            for jc in JUNCTIONS:
                # 時刻 t におけるフェーズを決定
                cum_t = 0
                target_phase = 11
                d_list = durations[jc]
                for p in range(12):
                    if t < cum_t + d_list[p]:
                        target_phase = p
                        break
                    cum_t += d_list[p]
                
                traci.trafficlight.setPhase(jc, target_phase)
            
            traci.simulationStep()
        
        self.episode_step += CYCLE_TIME

        # 3. 観測と報酬計算
        obs = self._get_observation()
        
        # 報酬計算
        current_waiting_times = {jc: self._get_junction_waiting_time(jc) for jc in JUNCTIONS}
        total_reward = 0
        
        rewards_info = {}

        for jc in JUNCTIONS:
            prev = self.prev_waiting_time[jc]
            curr = current_waiting_times[jc]
            
            # 隣接交差点の変化率平均
            adj_changes = []
            for adj in NEIGHBORS[jc]:
                adj_prev = self.prev_waiting_time[adj]
                adj_curr = current_waiting_times[adj]
                # 変化量 (正なら悪化)
                adj_changes.append(adj_curr - adj_prev)
            
            avg_adj_change = np.mean(adj_changes) if adj_changes else 0
            
            # 減少率計算
            # ReductionRate = (前回 - 今回) - α(隣接変化平均) / 前回
            # 前回が0の場合は分母を1として扱う（エラー回避）
            denom = prev if prev > 0 else 1.0
            
            # 式: 待ち時間が減ると (Prev - Curr) > 0。隣接が増えると (adj_change > 0) -> マイナス項でペナルティ
            reduction_rate = ((prev - curr) - ALPHA * avg_adj_change) / denom
            
            # 報酬テーブル
            reward = 0.0
            if reduction_rate >= 0.20:
                reward = 2.0  # 大幅改善
            elif 0.05 <= reduction_rate < 0.20:
                reward = 1.0  # 改善
            elif -0.05 <= reduction_rate < 0.05:
                reward = 0.0  # 維持
            elif -0.20 <= reduction_rate < -0.05:
                reward = -1.0 # 悪化
            else: # < -0.20
                reward = -3.0 # 大幅悪化
            
            total_reward += reward
            rewards_info[jc] = reward
            
            # 記録更新
            self.total_waiting_time_history[jc].append(curr)

        # 次ステップのために更新
        self.prev_waiting_time = current_waiting_times

        # 終了判定
        terminated = False
        truncated = False
        if self.episode_step >= EPISODE_LENGTH:
            terminated = True

        info = {"rewards": rewards_info}
        
        return obs, total_reward, terminated, truncated, info

    def _get_junction_waiting_time(self, junction_id):
        """交差点の総待ち時間（全流入レーンの合計）を取得"""
        wait_time = 0.0
        edges = INCOMING_EDGES[junction_id]
        for edge in edges:
            for lane_idx in range(3): # 3レーンと仮定
                lane_id = f"{edge}_{lane_idx}"
                # 存在確認（念のため）
                try:
                    wait_time += traci.lane.getWaitingTime(lane_id)
                except:
                    pass
        return wait_time

    def _get_observation(self):
        """全交差点の状態（72次元×4）を取得して結合"""
        full_obs = []
        
        # 全交差点の基本特徴量を取得
        junction_features = {}
        for jc in JUNCTIONS:
            feats = []
            edges = INCOMING_EDGES[jc]
            for edge in edges:
                for lane_idx in range(3):
                    lane_id = f"{edge}_{lane_idx}"
                    try:
                        halting = traci.lane.getLastStepHaltingNumber(lane_id)
                        waiting_time = traci.lane.getWaitingTime(lane_id)
                    except:
                        halting = 0
                        waiting_time = 0
                    feats.append(halting)
                    feats.append(waiting_time)
            # feats size: 4 edges * 3 lanes * 2 metrics = 24
            junction_features[jc] = feats

        # 各交差点ごとに、自分＋隣接の情報を結合 (72次元)
        for jc in JUNCTIONS:
            # 自分の状態 (24)
            obs_part = list(junction_features[jc])
            
            # 隣接の状態 (24 * 2)
            # 隣接順序はNEIGHBORS定義に従う
            for adj in NEIGHBORS[jc]:
                obs_part.extend(junction_features[adj])
            
            full_obs.extend(obs_part)
            
        return np.array(full_obs, dtype=np.float32)

    def close(self):
        traci.close()

# ==========================================
# イプシロン減衰スケジュール用のコールバック
# ==========================================
class EpsilonDecayCallback(BaseCallback):
    def __init__(self, total_timesteps, start_eps=1.0, end_eps=0.05, decay_fraction=0.8, verbose=0):
        super(EpsilonDecayCallback, self).__init__(verbose)
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_steps = total_timesteps * decay_fraction

    def _on_step(self) -> bool:
        # 現在の進捗率
        fraction = min(1.0, float(self.num_timesteps) / float(self.decay_steps))
        # 線形減衰
        current_eps = self.start_eps + fraction * (self.end_eps - self.start_eps)
        
        # モデルの探索率を更新
        self.model.exploration_rate = current_eps
        return True

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    # 環境作成
    env = SumoTrafficEnv()

    # パラメータ設定
    steps_per_episode = EPISODE_LENGTH // CONTROL_STEP # 5400 / 200 = 27
    total_timesteps = steps_per_episode * TOTAL_EPISODES

    # モデル定義 (DQN)
    # exploration_fraction等はCallbackで手動制御あるいはパラメータで指定可能だが、
    # SB3のDQNはパラメータでスケジュール可能
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=1e-3, 
        buffer_size=10000,
        learning_starts=100, 
        batch_size=32, 
        gamma=0.99,
        train_freq=1, 
        gradient_steps=1,
        target_update_interval=10,
        exploration_fraction=0.8,    # 全体の80%で減衰
        exploration_initial_eps=1.0, # 開始 100%
        exploration_final_eps=0.05,  # 終了 5%
        verbose=1
    )

    print("Training Started...")
    model.learn(total_timesteps=total_timesteps)
    print("Training Finished.")

    # 履歴データの取得（最後の環境の状態から履歴を取り出すハック）
    history = env.total_waiting_time_history
    
    # プロット
    plt.figure(figsize=(10, 6))
    steps = range(1, len(history["C1"]) + 1)
    
    for jc in JUNCTIONS:
        plt.plot(steps, history[jc], label=f"Junction {jc}")
    
    plt.xlabel(f"Steps (x{CYCLE_TIME} sec)")
    plt.ylabel("Total Waiting Time (sec)")
    plt.title("Waiting Time per Junction over Episodes")
    plt.legend()
    plt.grid(True)
    
    # 画像保存または表示
    plt.savefig("waiting_time_result.png")
    # plt.show() # 環境によっては表示できないため保存推奨
    print("Result graph saved as waiting_time_result.png")

    env.close()