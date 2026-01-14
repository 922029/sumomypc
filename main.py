import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# --- SUMO_HOMEの設定 ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
import traci

# ==========================================
# 1. 実験設定（ここを切り替えて比較を行う）
# ==========================================
# モード設定: 'independent' (独立), 'neighbor' (近傍), 'all' (全域)
LEARNING_MODE = 'independent' 

# ==========================================
# 2. 変更してはいけない共通定数（比較の公平性のため）
# ==========================================
# [COMMON_START]
GAMMA = 0.95            # 割引率
LEARNING_RATE = 0.001   # 学習率
MEMORY_SIZE = 10000     # リプレイバッファ
BATCH_SIZE = 64         # バッチサイズ
CYCLE_TIME = 90         # 1サイクルの合計時間
MIN_GREEN = 10          # 最小青時間
YELLOW_TIME = 3         # 黄信号の時間
# [COMMON_END]

TLS_IDS = ['C1', 'C2', 'C3', 'C4']
NEIGHBORS = {
    'C1': ['C2', 'C3'],
    'C2': ['C1', 'C4'],
    'C3': ['C1', 'C4'],
    'C4': ['C2', 'C3']
}

# --- DQNモデルの定義 ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        # [MODEL_ARCH_STRICT] ネットワーク構造は固定
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q = self.model(states).gather(1, actions)
        next_q = self.model(next_states).max(1)[0].detach().unsqueeze(1)
        target_q = rewards + (GAMMA * next_q * (1 - dones))

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- SUMO環境制御クラス ---
class SumoEnvironment:
    def __init__(self, mode):
        self.mode = mode
        self.tls_ids = TLS_IDS
        # 行動空間: (南北直進, 南北右折)
        self.action_list = []
        for ns_g in range(10, 51, 10):
            for ns_r in range(10, 31, 10):
                self.action_list.append((ns_g, ns_r))
        self.action_dim = len(self.action_list)

    def get_tls_state(self, tls_id):
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = sorted(list(set(lanes))) 
        halting = [traci.lane.getLastStepHaltingNumber(l) for l in unique_lanes]
        waiting = [traci.lane.getWaitingTime(l) for l in unique_lanes]
        return halting + waiting

    def get_state_by_mode(self, tls_id):
        all_data = {t: self.get_tls_state(t) for t in self.tls_ids}
        if self.mode == 'independent':
            return np.array(all_data[tls_id])
        elif self.mode == 'neighbor':
            state = list(all_data[tls_id])
            for n in NEIGHBORS[tls_id]:
                state += all_data[n]
            return np.array(state)
        elif self.mode == 'all':
            state = []
            for t in self.tls_ids:
                state += all_data[t]
            return np.array(state)

    def set_tls_durations(self, tls_id, action_idx):
        ns_g, ns_r = self.action_list[action_idx]
        # 東西時間はサイクルから引いて算出
        # (南北直進 + 黄 + 南北右折 + 黄 + 東西直進 + 黄) = 90
        ew_g = CYCLE_TIME - (ns_g + ns_r + (YELLOW_TIME * 3)) 
        ew_g = max(ew_g, MIN_GREEN)

        # 実際の設定に合わせたフェーズ時間の上書き
        # 注意: net.xmlの構成に依存します。以下は概念的な適用例
        durations = [ns_g, YELLOW_TIME, ns_r, YELLOW_TIME, ew_g, YELLOW_TIME]
        # ここでは単純化のため、シミュレーション内でこの順序で進むことを期待
        return durations

    def get_system_halting(self):
        # [REWARD_BASE] 報酬の基礎となる全車線の停止車両数合計
        total = 0
        for tls in self.tls_ids:
            lanes = set(traci.trafficlight.getControlledLanes(tls))
            for l in lanes:
                total += traci.lane.getLastStepHaltingNumber(l)
        return total

# --- メイン学習ループ ---
def run_experiment(num_episodes=100):
    env = SumoEnvironment(LEARNING_MODE)
    
    # --- [FIX] エラー対策: 起動前に状態取得できないため、一度仮起動して次元を確認 ---
    traci.start(["sumo", "-c", "4cross.sumocfg", "--no-warnings"])
    sample_state = env.get_state_by_mode('C1')
    state_dim = len(sample_state)
    traci.close()
    # --------------------------------------------------------------------

    agents = {tls: DQNAgent(state_dim, env.action_dim) for tls in TLS_IDS}
    history_wait_episodes = [] # エピソードごとの平均待ち時間を記録

    for ep in range(num_episodes):
        traci.start(["sumo", "-c", "4cross.sumocfg", "--no-warnings"])
        
        step = 0
        ep_halting_logs = []
        prev_halting = env.get_system_halting()
        
        while traci.simulation.getMinExpectedNumber() > 0 and step < 3600:
            states = {tls: env.get_state_by_mode(tls) for tls in TLS_IDS}
            actions = {tls: agents[tls].select_action(states[tls]) for tls in TLS_IDS}
            
            # アクション適用（1サイクル進める）
            traci.simulationStep(step + CYCLE_TIME)
            step += CYCLE_TIME
            
            current_halting = env.get_system_halting()
            
            # [REWARD_LOGIC] 差分が短くなれば正の報酬、増えれば負の報酬
            reward = (prev_halting - current_halting) * 1.0 
            
            next_states = {tls: env.get_state_by_mode(tls) for tls in TLS_IDS}
            
            for tls in TLS_IDS:
                agents[tls].memory.append((states[tls], actions[tls], reward, next_states[tls], False))
                agents[tls].replay()
            
            prev_halting = current_halting
            ep_halting_logs.append(current_halting)

        avg_ep_halting = np.mean(ep_halting_logs) if ep_halting_logs else 0
        history_wait_episodes.append(avg_ep_halting)
        
        print(f"Episode {ep+1}/{num_episodes} - Mode: {LEARNING_MODE} - Avg Halting: {avg_ep_halting:.2f} - Epsilon: {agents['C1'].epsilon:.3f}")
        traci.close()

    # --- 10エピソードごとのグラフ出力 ---
    if len(history_wait_episodes) >= 10:
        indices = range(10, num_episodes + 1, 10)
        averages = [np.mean(history_wait_episodes[i-10:i]) for i in indices]
        
        plt.figure(figsize=(10, 6))
        plt.plot(indices, averages, marker='o', label=LEARNING_MODE)
        plt.title(f'Learning Progress (10-Episode Average)')
        plt.xlabel('Episode')
        plt.ylabel('Average Halting Vehicles')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'result_{LEARNING_MODE}.png')
        plt.show()

if __name__ == "__main__":
    run_experiment(num_episodes=100)