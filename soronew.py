import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
import itertools

# ==========================================
# 設定と定数
# ==========================================
SUMOCFG_FILE = "4cross.sumocfg"
EPISODES = 10
SIMULATION_TIME = 5400
CYCLE_TIME = 200

# 信号機の設定
FIXED_YELLOW_TIME = 3
FIXED_ALL_RED_TIME = 2
FIXED_LOST_PER_PHASE = FIXED_YELLOW_TIME + FIXED_ALL_RED_TIME
TOTAL_PHASES = 4
TOTAL_LOST_TIME = FIXED_LOST_PER_PHASE * TOTAL_PHASES

ALLOCATABLE_TIME = CYCLE_TIME - TOTAL_LOST_TIME
TIME_UNIT = 15
TOTAL_BLOCKS = ALLOCATABLE_TIME // TIME_UNIT

TL_IDS = ["C1", "C2", "C3", "C4"]

INCOMING_EDGES = {
    "C1": ["N1_C1", "W1_C1", "C2_C1", "C3_C1"],
    "C2": ["N2_C2", "C1_C2", "E1_C2", "C4_C2"],
    "C3": ["C1_C3", "W2_C3", "C4_C3", "S1_C3"],
    "C4": ["C2_C4", "C3_C4", "E2_C4", "S2_C4"],
}

PHASE_STATES = [
    "GGgrrrGGgrrr", # 0: Action0 (North-South Straight/Left)
    "yyyrrryyyrrr", # 1: Yellow
    "rrrrrrrrrrrr", # 2: All Red
    "rrGrrrrrGrrr", # 3: Action1 (North-South Right)
    "rryrrrrryrrr", # 4: Yellow
    "rrrrrrrrrrrr", # 5: All Red
    "rrrGGgrrrGGg", # 6: Action2 (East-West Straight/Left)
    "rrryyyrrryyy", # 7: Yellow
    "rrrrrrrrrrrr", # 8: All Red
    "rrrrrGrrrrrG", # 9: Action3 (East-West Right)
    "rrrrryrrrrry", # 10: Yellow
    "rrrrrrrrrrrr"  # 11: All Red
]

# ==========================================
# 行動空間生成
# ==========================================
def generate_valid_allocations():
    valid_actions = []
    for p0 in range(1, TOTAL_BLOCKS - 2):
        for p1 in range(1, TOTAL_BLOCKS - 2):
            for p2 in range(1, TOTAL_BLOCKS - 2):
                for p3 in range(1, TOTAL_BLOCKS - 2):
                    if p0 + p1 + p2 + p3 == TOTAL_BLOCKS:
                        times = [p0*TIME_UNIT, p1*TIME_UNIT, p2*TIME_UNIT, p3*TIME_UNIT]
                        valid_actions.append(times)
    return valid_actions

VALID_ACTIONS = generate_valid_allocations()
NUM_ACTIONS = len(VALID_ACTIONS)

# ==========================================
# 環境クラス定義
# ==========================================
class SumoIndependentEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=99999, shape=(24,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        
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
            '--time-to-teleport', '-1'
        ]
        
        self.is_running = False
        self.episode_step = 0
        self.prev_cycle_cost = {tl: 0.0 for tl in TL_IDS}

    def reset(self, seed=None, options=None):
        if self.is_running:
            traci.close()
        traci.start(self.sumo_cmd)
        self.is_running = True
        self.episode_step = 0
        self.prev_cycle_cost = {tl: 0.0 for tl in TL_IDS}
        return self._get_observations()

    def step(self, actions_dict):
        self.episode_step += 1
        current_green_times = {tl: VALID_ACTIONS[idx] for tl, idx in actions_dict.items()}
        current_cycle_cost = {tl: 0.0 for tl in TL_IDS}
        
        # フェーズスケジュールの組み立て
        phase_schedule = {}
        for tl_id in TL_IDS:
            times = current_green_times[tl_id]
            phase_schedule[tl_id] = [
                (times[0], 0), (3, 1), (2, 2),
                (times[1], 3), (3, 4), (2, 5),
                (times[2], 6), (3, 7), (2, 8),
                (times[3], 9), (3, 10), (2, 11)
            ]

        current_phase_idx = {tl: 0 for tl in TL_IDS}
        current_phase_time = {tl: 0 for tl in TL_IDS}
        
        for tl_id in TL_IDS:
            traci.trafficlight.setRedYellowGreenState(tl_id, PHASE_STATES[phase_schedule[tl_id][0][1]])

        # サイクル実行
        for _ in range(CYCLE_TIME):
            traci.simulationStep()
            for tl_id in TL_IDS:
                halt_sum = 0
                for edge in INCOMING_EDGES[tl_id]:
                    for i in range(3):
                        halt_sum += traci.lane.getLastStepHaltingNumber(f"{edge}_{i}")
                current_cycle_cost[tl_id] += halt_sum

            # 信号制御の更新
            for tl_id in TL_IDS:
                current_phase_time[tl_id] += 1
                p_idx = current_phase_idx[tl_id]
                if current_phase_time[tl_id] >= phase_schedule[tl_id][p_idx][0]:
                    current_phase_time[tl_id] = 0
                    current_phase_idx[tl_id] += 1
                    if current_phase_idx[tl_id] < len(phase_schedule[tl_id]):
                        state = PHASE_STATES[phase_schedule[tl_id][current_phase_idx[tl_id]][1]]
                        traci.trafficlight.setRedYellowGreenState(tl_id, state)

        # 報酬計算 (変化の割合に基づくテーブル)
        rewards = {}
        next_obs = self._get_observations()
        
        for tl_id in TL_IDS:
            prev = self.prev_cycle_cost[tl_id]
            now = current_cycle_cost[tl_id]
            
            # 初回サイクルは基準がないため報酬0
            if self.episode_step == 1 or prev == 0:
                reward = 0.0
            else:
                # 減少率を計算 (正の値なら渋滞が減った)
                reduction_rate = (prev - now) / prev
                
                # 指定の報酬テーブル
                if reduction_rate >= 0.20:
                    reward = 2.0  # 大幅改善
                elif reduction_rate >= 0.05:
                    reward = 1.0  # 改善
                elif reduction_rate >= -0.05:
                    reward = 0.0  # 維持
                elif reduction_rate >= -0.20:
                    reward = -1.0 # 悪化
                else:
                    reward = -3.0 # 大幅悪化
            
            rewards[tl_id] = reward
            self.prev_cycle_cost[tl_id] = now

        done = traci.simulation.getTime() >= SIMULATION_TIME
        return next_obs, rewards, {tl: done for tl in TL_IDS}, {tl: {"wait_cost": current_cycle_cost[tl]} for tl in TL_IDS}
    
    def _get_observations(self):
        observations = {}
        for tl_id in TL_IDS:
            obs_list = []
            for edge in INCOMING_EDGES[tl_id]:
                for i in range(3):
                    lane_id = f"{edge}_{i}"
                    try:
                        obs_list.extend([traci.lane.getLastStepHaltingNumber(lane_id), traci.lane.getWaitingTime(lane_id)])
                    except:
                        obs_list.extend([0, 0])
            observations[tl_id] = np.array(obs_list, dtype=np.float32)
        return observations
        
    def close(self):
        traci.close()

# ==========================================
# メイン学習ループ
# ==========================================
if __name__ == "__main__":
    env = SumoIndependentEnv()
    agents = {}
    for tl_id in TL_IDS:
        agents[tl_id] = DQN("MlpPolicy", env, verbose=0, target_update_interval=50)
        log_path = f"./logs/{tl_id}"
        agents[tl_id].set_logger(configure(log_path, ["stdout", "csv"]))
        agents[tl_id]._current_progress_remaining = 1.0

    all_episode_wait_times = {tl: [] for tl in TL_IDS}
    
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        total_rewards = {tl: 0 for tl in TL_IDS}
        total_wait_costs = {tl: 0 for tl in TL_IDS}
        steps = 0
        
        while not done:
            actions = {tl: int(agents[tl].predict(obs[tl], deterministic=False)[0]) for tl in TL_IDS}
            next_obs, rewards, dones, infos = env.step(actions)
            
            for tl_id in TL_IDS:
                agents[tl_id].replay_buffer.add(
                    obs[tl_id].reshape(1, -1), next_obs[tl_id].reshape(1, -1),
                    np.array([actions[tl_id]]), np.array([rewards[tl_id]]),
                    np.array([dones[tl_id]]), [infos[tl_id]]
                )
                agents[tl_id].train(gradient_steps=1, batch_size=32)
                total_rewards[tl_id] += rewards[tl_id]
                total_wait_costs[tl_id] += infos[tl_id]["wait_cost"]
            
            obs = next_obs
            done = all(dones.values())
            steps += 1
            
        print(f"Episode {ep+1}/{EPISODES} Finished.")
        for tl_id in TL_IDS:
            avg_wait = total_wait_costs[tl_id] / steps
            all_episode_wait_times[tl_id].append(avg_wait)

    env.close()
    
    plt.figure(figsize=(10, 6))
    for tl_id in TL_IDS:
        plt.plot(range(1, EPISODES + 1), all_episode_wait_times[tl_id], label=tl_id, marker='o')
    plt.title("Avg Waiting Cost per Cycle")
    plt.xlabel("Episode")
    plt.ylabel("Avg Waiting Cost")
    plt.legend()
    plt.grid(True)
    plt.savefig("waiting_time_result.png")