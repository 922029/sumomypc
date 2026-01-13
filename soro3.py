import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# SUMO_HOMEの設定
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.path.append('/usr/share/sumo/tools')

import traci
import sumolib

class SumoTrafficEnvIQL:
    def __init__(self, config_file="4cross.sumocfg", use_gui=False):
        self.config_file = config_file
        self.use_gui = use_gui
        self.sumo_binary = sumolib.checkBinary('sumo-gui' if use_gui else 'sumo')
        self.tls_ids = ['C1', 'C2', 'C3', 'C4']
        
        self.neighbors = {
            'C1': ['C2', 'C3'],
            'C2': ['C1', 'C4'],
            'C3': ['C1', 'C4'],
            'C4': ['C2', 'C3']
        }
        
        self.cycle_time = 90
        self.yellow_time = 2
        self.red_time = 3
        self.right_turn_time = 6
        self.min_green = 5
        
        self.fixed_time_total = (self.yellow_time + self.red_time + self.right_turn_time) * 2
        self.available_green_total = self.cycle_time - self.fixed_time_total # 68秒
        
        self.possible_ns_greens = np.arange(self.min_green, self.available_green_total - self.min_green + 1, 5)
        self.action_space_size = len(self.possible_ns_greens)
        
        self.tls_lanes = {}
        self.prev_waiting_times = {tls: 0 for tls in self.tls_ids}

    def start_simulation(self):
        try:
            traci.close()
        except:
            pass
        traci.start([self.sumo_binary, "-c", self.config_file, "--tripinfo-output", "tripinfo.xml", "--no-warnings"])
        for tls in self.tls_ids:
            self.tls_lanes[tls] = traci.trafficlight.getControlledLanes(tls)
        self.prev_waiting_times = {tls: 0 for tls in self.tls_ids}

    def _get_single_tls_state(self, tls_id):
        lanes = self.tls_lanes[tls_id]
        halting = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes])
        # 状態空間を圧縮 (5段階)
        s_halting = min(halting // 10, 4) 
        return int(s_halting)

    def get_state(self):
        base_states = {tls: self._get_single_tls_state(tls) for tls in self.tls_ids}
        states = {}
        for tls in self.tls_ids:
            combined_state = [base_states[tls]]
            for nb in self.neighbors[tls]:
                combined_state.append(base_states[nb])
            states[tls] = tuple(combined_state)
        return states

    def step(self, actions):
        for tls_id, action_idx in actions.items():
            ns_green = self.possible_ns_greens[action_idx]
            ew_green = self.available_green_total - ns_green
            
            logics = traci.trafficlight.getAllProgramLogics(tls_id)
            if not logics: continue
            
            logic = logics[0]
            phases = list(logic.phases)
            
            if len(phases) >= 8:
                phases[0].duration = float(ns_green)
                phases[1].duration = float(self.yellow_time)
                phases[2].duration = float(self.red_time)
                phases[3].duration = float(self.right_turn_time)
                phases[4].duration = float(ew_green)
                phases[5].duration = float(self.yellow_time)
                phases[6].duration = float(self.red_time)
                phases[7].duration = float(self.right_turn_time)
                
                logic.phases = tuple(phases)
                traci.trafficlight.setProgramLogic(tls_id, logic)

        for _ in range(self.cycle_time):
            traci.simulationStep()
        
        next_states = self.get_state()
        avg_wait_this_step = []
        rewards = {}
        
        for tls in self.tls_ids:
            lanes = self.tls_lanes[tls]
            current_wait = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes])
            
            # 報酬: 改善度 + 渋滞量ペナルティ
            diff_reward = self.prev_waiting_times[tls] - current_wait
            absolute_penalty = - (current_wait / 10.0)
            
            rewards[tls] = float(diff_reward + absolute_penalty)
            self.prev_waiting_times[tls] = current_wait
            avg_wait_this_step.append(current_wait)
            
        return next_states, rewards, np.mean(avg_wait_this_step)

    def close(self):
        try:
            traci.close()
        except:
            pass

class QLearningAgent:
    def __init__(self, action_size, learning_rate=0.05, discount_factor=0.95, epsilon=1.0):
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon        # 121行目: 初期値を1.0に変更 (最初は100%探索)
        self.epsilon_decay = 0.98    # 122行目: 減衰率を追加
        self.epsilon_min = 0.05      # 123行目: 最小探索率を追加
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (target - predict)
    
    def decay_epsilon(self):
        # 137行目付近: 新規追加。エピソード毎に呼び出し
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train():
    env = SumoTrafficEnvIQL(config_file="4cross.sumocfg", use_gui=False)
    agents = {tls: QLearningAgent(env.action_space_size) for tls in env.tls_ids}
    
    episodes = 3
    history_waiting_times = []

    print(f"Starting training for {episodes} episodes...")

    try:
        for e in range(episodes):
            env.start_simulation()
            states = env.get_state()
            total_waiting = []
            
            for _ in range(40):
                actions = {tls: agents[tls].get_action(states[tls]) for tls in env.tls_ids}
                next_states, rewards, step_wait = env.step(actions)
                
                for tls in env.tls_ids:
                    agents[tls].learn(states[tls], actions[tls], rewards[tls], next_states[tls])
                
                states = next_states
                total_waiting.append(step_wait)
            
            # 168行目: エピソード終了後に探索率を減少させる
            for tls in env.tls_ids:
                agents[tls].decay_epsilon()
                
            avg_wait_episode = np.mean(total_waiting)
            history_waiting_times.append(avg_wait_episode)
            
            if (e + 1) % 5 == 0:
                eps = agents['C1'].epsilon
                print(f"Episode {e+1}/{episodes} - Wait: {avg_wait_episode:.2f}, Epsilon: {eps:.3f}")
            
            env.close()
    except Exception as ex:
        print(f"An error occurred: {ex}")
        env.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history_waiting_times, marker='o', linestyle='-', color='g')
    plt.title("Improved Neighbor-Aware IQL (with Epsilon Decay)")
    plt.xlabel("Episode")
    plt.ylabel("Average Waiting Time (Halting Vehicles)")
    plt.grid(True)
    plt.savefig("learning_curve_neighbor.png")
    plt.show()

if __name__ == "__main__":
    train()