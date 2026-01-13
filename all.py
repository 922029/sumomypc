import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# SUMO_HOMEの設定
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
import traci
import sumolib

# --- 定数設定 ---
CYCLE_TIME = 90
MAX_AVAILABLE_GREEN = 68 
GREEN_CHOICES = [10, 20, 30, 40, 50, 60] 
NUM_ACTIONS_PER_TLS = len(GREEN_CHOICES)
NUM_TLS = 4
STATE_DIM = 96  # (12車線 * 2情報:Halting, Waiting) * 4交差点

# 4交差点の同時アクションインデックス (簡易化のため独立制御を前提とした設計も可能ですが、ここでは集中制御)
TOTAL_JOINT_ACTIONS = NUM_ACTIONS_PER_TLS ** NUM_TLS 

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class IntegratedSumoEnv:
    def __init__(self, config_file="4cross.sumocfg", use_gui=False):
        self.sumo_binary = sumolib.checkBinary('sumo-gui' if use_gui else 'sumo')
        self.config_file = config_file
        self.tls_ids = ['C1', 'C2', 'C3', 'C4']
        self.prev_waiting_times = {tls: 0 for tls in self.tls_ids}
        self.tls_lanes = {}

    def start(self):
        traci.start([self.sumo_binary, "-c", self.config_file, "--waiting-time-memory", "100"])
        for tls in self.tls_ids:
            all_lanes = list(set(traci.trafficlight.getControlledLanes(tls)))
            # net.xmlの構成に基づき、接続されている12の入線を取得
            self.tls_lanes[tls] = [l for l in all_lanes if "_" in l][:12]

    def reset(self):
        if traci.isLoaded():
            traci.close()
        self.start()
        self.prev_waiting_times = {tls: 0 for tls in self.tls_ids}
        return self._get_state()

    def _get_state(self):
        state_vector = []
        for tls in self.tls_ids:
            for lane in self.tls_lanes[tls]:
                halting = traci.lane.getLastStepHaltingNumber(lane)
                waiting = traci.lane.getWaitingTime(lane)
                state_vector.append(halting / 20.0)
                state_vector.append(waiting / 100.0)
        return np.array(state_vector, dtype=np.float32)

    def _get_avg_waiting_time(self):
        """現在の全交差点の車両1台あたりの平均待ち時間を計算"""
        total_wait = 0
        total_vehicles = 0
        for tls in self.tls_ids:
            for lane in self.tls_lanes[tls]:
                total_wait += traci.lane.getWaitingTime(lane)
                total_vehicles += traci.lane.getLastStepVehicleNumber(lane)
        return total_wait / max(1, total_vehicles)

    def step(self, joint_action_idx):
        # アクションのデコード
        actions = []
        temp = joint_action_idx
        for _ in range(NUM_TLS):
            actions.append(GREEN_CHOICES[temp % NUM_ACTIONS_PER_TLS])
            temp //= NUM_ACTIONS_PER_TLS

        # 信号プログラムの動的変更（南北青時間を設定）
        for i, tls in enumerate(self.tls_ids):
            ns_green = actions[i]
            # フェーズ定義に合わせて調整 (例: Phase 0が南北青, Phase 4が東西青とする)
            traci.trafficlight.setPhaseDuration(tls, ns_green) # 厳密にはsetPhaseと併用が必要
            
        # 1サイクル分シミュレートし、その間の平均待ち時間をサンプリング
        cycle_wait_accumulator = []
        for _ in range(CYCLE_TIME):
            traci.simulationStep()
            if traci.simulation.getTime() % 10 == 0: # 10秒ごとに統計取得
                cycle_wait_accumulator.append(self._get_avg_waiting_time())

        next_state = self._get_state()
        reward = self._calculate_reward()
        done = traci.simulation.getTime() >= 3600
        
        avg_wait_in_cycle = np.mean(cycle_wait_accumulator) if cycle_wait_accumulator else 0
        return next_state, reward, done, avg_wait_in_cycle

    def _calculate_reward(self):
        total_reward = 0
        for tls in self.tls_ids:
            current_wait = sum([traci.lane.getWaitingTime(l) for l in self.tls_lanes[tls]])
            # 待ち時間が減ればプラス報酬
            reward = self.prev_waiting_times[tls] - current_wait
            self.prev_waiting_times[tls] = current_wait
            total_reward += reward
        return float(total_reward) / 100.0

    def close(self):
        traci.close()

def train():
    env = IntegratedSumoEnv(use_gui=False)
    # 状態空間 96, アクション空間 6^4=1296
    model = DQN(STATE_DIM, TOTAL_JOINT_ACTIONS)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    memory = deque(maxlen=2000)
    
    epsilon = 1.0
    batch_size = 64
    num_episodes = 3
    
    episode_rewards = []
    episode_waits = [] # 各エピソードの平均待ち時間を保存

    print("--- Starting Training ---")
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step_waits = []
        
        # 3600秒 / 90秒サイクル = 40ステップ
        for step in range(40):
            if random.random() < epsilon:
                action = random.randint(0, TOTAL_JOINT_ACTIONS - 1)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action = model(state_t).argmax().item()
            
            next_state, reward, done, avg_wait = env.step(action)
            memory.append((state, action, reward, next_state, done))
            
            state = next_state
            total_reward += reward
            step_waits.append(avg_wait)
            
            # 学習
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                s_b, a_b, r_b, ns_b, d_b = zip(*batch)
                
                s_b = torch.FloatTensor(np.array(s_b))
                a_b = torch.LongTensor(a_b).unsqueeze(1)
                r_b = torch.FloatTensor(r_b).unsqueeze(1)
                ns_b = torch.FloatTensor(np.array(ns_b))
                d_b = torch.FloatTensor(d_b).unsqueeze(1)
                
                q_values = model(s_b).gather(1, a_b)
                next_q_values = model(ns_b).max(1)[0].detach().unsqueeze(1)
                target_q = r_b + (0.98 * next_q_values * (1 - d_b))
                
                loss = criterion(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done: break
            
        epsilon = max(0.1, epsilon * 0.96)
        
        avg_episode_wait = np.mean(step_waits)
        episode_rewards.append(total_reward)
        episode_waits.append(avg_episode_wait)
        
        print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f} | Avg Wait: {avg_episode_wait:.2f}s | Epsilon: {epsilon:.2f}")

    env.close()

    # --- グラフ出力 ---


    plt.subplot(1, 2, 2)
    plt.plot(episode_waits, color='red')
    plt.title('Global Average Waiting Time (All Junctions)')
    plt.xlabel('Episode')
    plt.ylabel('Wait Time (s)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()