import os, sys
import numpy as np

# SUMO_HOMEの設定
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    # デフォルトパスが必要な場合はここに追加してください
    pass

try:
    import traci
    import sumolib
except ImportError:
    print("Error: TraCI or SUMO not found.")

class SumoTrafficEnv:
    """
    4つの交差点（C1, C2, C3, C4）を個別に制御するSUMO環境。
    仕様：
    - 状態：流入12道路の停止車両数(Halting Number)と待ち時間(Waiting Time)
    - 行動：南北方向の青信号時間の設定
    - 報酬：待ち時間の増減（前回の待ち時間 - 今回の待ち時間）
    """
    def __init__(self, config_file="4cross.sumocfg", sim_max_time=3600, use_gui=False):
        self.config_file = config_file
        self.sim_max_time = sim_max_time
        self.use_gui = use_gui
        self.sumo_binary = sumolib.checkBinary('sumo-gui' if use_gui else 'sumo')
        
        self.tls_ids = ['C1', 'C2', 'C3', 'C4']
        self.num_agents = len(self.tls_ids)
        
        # 流入レーンの定義（各交差点12本）
        self.tls_lanes = {
            'C1': ['N1_C1_0', 'N1_C1_1', 'N1_C1_2', 'C3_C1_0', 'C3_C1_1', 'C3_C1_2',
                   'C2_C1_0', 'C2_C1_1', 'C2_C1_2', 'W1_C1_0', 'W1_C1_1', 'W1_C1_2'],
            'C2': ['C1_C2_0', 'C1_C2_1', 'C1_C2_2', 'C4_C2_0', 'C4_C2_1', 'C4_C2_2',
                   'E1_C2_0', 'E1_C2_1', 'E1_C2_2', 'N2_C2_0', 'N2_C2_1', 'N2_C2_2'], 
            'C3': ['C1_C3_0', 'C1_C3_1', 'C1_C3_2', 'C4_C3_0', 'C4_C3_1', 'C4_C3_2',
                   'S1_C3_0', 'S1_C3_1', 'S1_C3_2', 'W2_C3_0', 'W2_C3_1', 'W2_C3_2'], 
            'C4': ['C2_C4_0', 'C2_C4_1', 'C2_C4_2', 'C3_C4_0', 'C3_C4_1', 'C3_C4_2',
                   'E2_C4_0', 'E2_C4_1', 'E2_C4_2', 'S2_C4_0', 'S2_C4_1', 'S2_C4_2'],
        }
        
        self.current_sim_time = 0
        self.cycle_time = 90.0
        
        # 信号時間の定義 (修正箇所)
        self.yellow_time = 2.0   # 黄色
        self.turn_time = 5.0     # 右折矢印
        self.all_red_time = 3.0  # 全赤（安全マージン）
        
        # RLで制御可能な最小・最大直進青時間
        # (90秒 - (黄2+右5+赤3)*2 = 70秒 を南北と東西で分ける)
        self.min_green = 10.0
        self.max_green = 70.0
        # 前回の待ち時間を記録する変数（報酬計算用）
        self.prev_waiting_times = {tls: 0.0 for tls in self.tls_ids}

    def reset(self):
        try:
            traci.close()
        except:
            pass
            
        traci.start([self.sumo_binary, "-c", self.config_file, "--no-warnings", "true"])
        
        self.current_sim_time = 0
        self.prev_waiting_times = {tls: 0.0 for tls in self.tls_ids}
        
        return self._get_state()

    def step(self, actions):
        """
        actions: 各交差点の南北青信号時間 [C1_ns, C2_ns, C3_ns, C4_ns]
        """
        # 1. 各信号機に行動を適用
        for i, tls_id in enumerate(self.tls_ids):
            ns_green = np.clip(actions[i], self.min_green, self.max_green)
            self._apply_traffic_logic(tls_id, ns_green)
            
        # 2. 1サイクル（90秒）分シミュレーションを進める
        for _ in range(int(self.cycle_time)):
            traci.simulationStep()
            self.current_sim_time += 1
            
        next_state = self._get_state()
        rewards = self._calculate_rewards()
        done = self.current_sim_time >= self.sim_max_time
        
        return next_state, rewards, done, {}

    def _apply_traffic_logic(self, tls_id, ns_green):
        """
        修正後のロジック:
        1. 南北直進(ns_green) -> 2. 黄色(3s) -> 3. 南北右折(6s) -> 4. 黄色(3s) -> 5. 全赤(2s)
        6. 東西直進(ew_green) -> 7. 黄色(3s) -> 8. 東西右折(6s) -> 9. 黄色(3s) -> 10. 全赤(2s)
        """
        fixed_overhead = (self.yellow_time * 2 + self.turn_time + self.all_red_time) * 2
        ew_green = self.cycle_time - ns_green - fixed_overhead
        ew_green = max(ew_green, 5.0) 

        # 状態文字列の例 (12〜16レーン想定)
        # G:青, g:優先度の低い青, y:黄, r:赤, s:停止
        # 右折矢印はインデックスに合わせて 'G' を設定してください
        
        logic = traci.trafficlight.Logic(
            programID="1", type=0, currentPhaseIndex=0,
            phases=[
                traci.trafficlight.Phase(ns_green, "GGGrrrrrrrrr"),        # 1. 南北直進
                traci.trafficlight.Phase(self.yellow_time, "yyyrrrrrrrrr"),   # 2. 南北黄
                traci.trafficlight.Phase(self.turn_time, "rrrGGGrrrrrr"),     # 3. 南北右折
                traci.trafficlight.Phase(self.yellow_time, "rrryyyrrrrrr"),   # 4. 黄
                traci.trafficlight.Phase(self.all_red_time, "rrrrrrrrrrrr"),  # 5. 全赤
                traci.trafficlight.Phase(ew_green, "rrrrrrGGGrrr"),        # 6. 東西直進
                traci.trafficlight.Phase(self.yellow_time, "rrrrrryyyrrr"),   # 7. 東西黄
                traci.trafficlight.Phase(self.turn_time, "rrrrrrrrrGGG"),     # 8. 東西右折 (※交差点構造に合わせる)
                traci.trafficlight.Phase(self.yellow_time, "rrrrrrrrryyy"),   # 9. 黄
                traci.trafficlight.Phase(self.all_red_time, "rrrrrrrrrrrr"),  # 10. 全赤
            ]
        )
        traci.trafficlight.setProgramLogic(tls_id, logic)


    def _get_state(self):
        """
        12本の流入道路の「停止車両数」と「待ち時間」を状態として返す
        """
        states = []
        for tls_id in self.tls_ids:
            lanes = self.tls_lanes[tls_id]
            tls_state = []
            for lane in lanes:
                # 停止車両数と待ち時間を取得
                halting = traci.lane.getLastStepHaltingNumber(lane)
                waiting = traci.lane.getWaitingTime(lane)
                
                # スケーリングして追加（学習効率のため）
                tls_state.append(halting / 20.0)
                tls_state.append(waiting / 100.0)
            states.append(tls_state) # 交差点ごとの24次元ベクトル
            
        return np.array(states, dtype=np.float32)

    def _calculate_rewards(self):
        """
        待ち時間が減ったらプラス、増えたらマイナスの報酬を計算。
        待ち時間定義 = traci.lane.getLastStepHaltingNumber() × 1秒
        """
        rewards = []
        for tls_id in self.tls_ids:
            lanes = self.tls_lanes[tls_id]
            
            # 現在の合計待ち時間（停止車両数 * 1s）
            current_wait = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes])
            
            # 報酬 = 前回 - 今回（正の値なら改善）
            reward = self.prev_waiting_times[tls_id] - current_wait
            
            # 履歴の更新
            self.prev_waiting_times[tls_id] = current_wait
            rewards.append(float(reward))
            
        return rewards

    def close(self):
        try:
            traci.close()
        except:
            pass

if __name__ == '__main__':
    # テスト実行コード
    env = SumoTrafficEnv(sim_max_time=300)
    try:
        obs = env.reset()
        # テスト行動：全交差点で南北30秒
        next_obs, rewards, done, _ = env.step([30.0, 30.0, 30.0, 30.0])
        print(f"ステップ完了。報酬: {rewards}")
    finally:
        env.close()