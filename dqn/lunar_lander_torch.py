# Authors: Matteo Ventali and Valerio Spagnoli
# ML Project : DQN for Lunar Lander environment

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import random as ran
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

window_size = 500

def function_plot(reward, epsilon_value):
    mean_mobile = np.convolve(reward, np.ones(window_size)/window_size, mode="valid")

    # Organizing the graph with two y axis
    fig, ax1 = plt.subplots()

    # First axis
    ax1.plot(mean_mobile, color='red', label='reward_greedy_policy')
    ax1.set_ylabel('epsilon greedy policy', color='red')
    ax1.set_xlabel('episodes', color='blue')

    # Second axis
    ax2 = ax1.twinx()
    ax2.plot(epsilon_value, color='orange', label='epsilon value')
    ax2.set_ylabel('epsilon value', color='orange')
    
    plt.title("Reward")
    plt.show()

def function_plot_combined(reward_eps, reward_random, epsilon_value):
    mean_mobile = np.convolve(reward_eps, np.ones(window_size)/window_size, mode="valid")
    mean_mobile_random = np.convolve(reward_random, np.ones(window_size)/window_size, mode="valid")

    # Organizing the graph with two y axis
    fig, ax1 = plt.subplots()

    # First axis
    ax1.plot(mean_mobile, color='red', label='reward_greedy_policy')
    ax1.plot(mean_mobile_random, color='green', label='reward_greedy_policy')
    ax1.set_ylabel('policies', color='red')
    ax1.set_xlabel('episodes', color='blue')

    # Second axis
    ax2 = ax1.twinx()
    ax2.plot(epsilon_value, color='orange', label='reward_greedy_policy')
    ax2.set_ylabel('epsilon value', color='orange')
    
    plt.title("Reward")
    plt.show()


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done) 
        self.buffer.append(data)

    def sample(self, batch_size):
        batch = ran.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)


class DQN:
    def __init__(self, state_dim, num_actions, device="cpu"):
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        ).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, batch):
        states = torch.tensor(np.array([s for s, q in batch]), dtype=torch.float32).to(self.device)
        q_targets = torch.tensor(np.array([q for s, q in batch]), dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        q_preds = self.model(states)
        loss = self.loss_fn(q_preds, q_targets)
        loss.backward()
        self.optimizer.step()

    def predict_qValue(self, state):
        state_input = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_input)
        return q_values.cpu().numpy()


class QLearner():
    def __init__(self, env: gym.Env, max_episodes=3500, gamma=0.99, alpha=0.1, end_eps=0.01, start_eps=1.0, eps_decay=0.999, model_name="dqn_model.pth"):
        self.env = env
        self.max_episodes = max_episodes        
        self.gamma = gamma
        self.alpha = alpha
        self.end_eps = end_eps
        self.eps = start_eps
        self.eps_decay = eps_decay
        self.batch_dimension = 64
        self.memory_capacity = 200000
        self.update_every = 4
        self.update_modality = 1 # 0 deterministic, 1 non determinstic
        self.model_name = "./dqn_models/" + model_name

    def _espilon_update(self):
        self.eps = max(self.eps_decay * self.eps, self.end_eps)

    def _next_action(self, modality, current_state, q_network : DQN):
        n = ran.random()
        if modality == 0 or n < self.eps: # Exploration
            return self.env.action_space.sample()
        else: # Exploitation
            q_values = q_network.predict_qValue(current_state)[0]
            a = np.argmax(q_values).item()
            return a

    def _prepareBatch(self, batch, q_network: DQN):
        training_set = [] # Result of preparing the data
        
        # Preparing the batch
        for t in batch: # (s, a, r, s', done)
            # Q(s,a) and Q(s',a) forall action a
            q_values_s = q_network.predict_qValue(t[0])[0]
            q_values_ns = q_network.predict_qValue(t[3])[0]
            
            # Q(s,a) = Q(s,a) + alpha[r + y*max_a'{Q(s',a')} - Q(s,a)]
            # Updating only in corrispondence of the action
            action = int(t[1])
            row = (t[0], q_values_s)
            if self.update_modality == 0:
                row[1][action] = t[2] + self.gamma * np.max(q_values_ns) * (1 - int(t[4]))
            else:
                row[1][action] = q_values_s[action] + self.alpha * (t[2] + self.gamma * np.max(q_values_ns) * (1 - int(t[4])) - q_values_s[action])
            
            training_set.append(row)

        return training_set

    def _normalize(self, state):
        amplitudes = [2.5, 2.5, 10., 10., 6.2831855, 10., 1., 1.]
        return state / amplitudes

    def DQN_Learning(self, modality=1): 
        self.eps = 1.0

        # Creation of the NN representing the Q-table
        q_network = DQN(8, self.env.action_space.n)

        # Creation of the dataset implementing the replay memory
        memory = ReplayBuffer(self.memory_capacity)

        # Starting of the environment
        s, _ = self.env.reset()

        # Array for collecting total rewards
        total_rewards = []
        eps_per_episode = []

        for n_episode in range(self.max_episodes):
            #print(f"Episode n: {n_episode}")
            episode_reward = 0
            done = False
            n_steps = 0
            while not done:
                # Select the action to be executed
                a = self._next_action(modality, s, q_network)

                # Execution of a
                ns, reward, terminated, truncated, _ = self.env.step(a)
                n_steps+=1
                episode_reward += reward
                done = terminated or truncated
                
                # (s,a,r,s',done) in replay buffer
                memory.add(self._normalize(s), a, reward, self._normalize(ns), done)
                #memory.add(s, a, reward, ns, done)

                # get a sample batch for training
                if ( len(memory) > self.batch_dimension and n_steps % self.update_every == 0 ):
                    batch = memory.sample(self.batch_dimension)
                    # Preparing the batch
                    training_set = self._prepareBatch(batch, q_network)
                    q_network.train(training_set)

                # Updating new state
                if not done:
                    s = ns
            
            # Stats of the episode
            eps_per_episode.append(self.eps)
            self._espilon_update()
            s, _ = self.env.reset()
            total_rewards.append(episode_reward)
            if n_episode > 50:
                print(f"(m={modality} episode {n_episode} {np.mean(total_rewards[-50:])} {self.eps})")

        if modality == 1:
            self._save_policy(q_network)
        return total_rewards, eps_per_episode

    def _save_policy(self, q_network: DQN):
        torch.save(q_network.model.state_dict(), self.model_name)

    def _load_policy(self, q_network: DQN):
        q_network.model.load_state_dict(torch.load(self.model_name))
        q_network.model.eval()

    def run_policy(self):
        q_network = DQN(8, self.env.action_space.n)
        self._load_policy(q_network)

        total_reward = 0
        episodes_reward = []
        n_episodes = 1000

        for i in range(n_episodes):
            s, _ = self.env.reset()
            terminated = truncated = False
            rw = 0
            while not (terminated or truncated):
                q_values = q_network.predict_qValue(s)[0]
                a = np.argmax(q_values)
                ns, r, terminated, truncated, _ = self.env.step(a)
                rw += r
                s = ns
            total_reward += rw
            episodes_reward.append(rw)
            print(f"({i}: rw={rw:.2f})")

        print(f"Mean Reward: {total_reward / n_episodes}")
        print(f"Mean Episode Reward: {np.mean(episodes_reward)}")


if __name__ == "__main__":
    #drive.mount('/content/drive')
    #output_dir = "/content/drive/MyDrive/reward_files_dqn/"
    #os.makedirs(output_dir, exist_ok=True)
    #os.makedirs("/content/drive/MyDrive/dqn_models/", exist_ok=True)
    
    # Lunar Lander Environment
    env = gym.make("LunarLander-v3", render_mode="human", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    
    # Menu
    mode = input("Select modality (0 = training, 1 = running): ").strip()
    model_file = input("File model (empty for default):").strip()

    # Learner object
    if model_file == "": # Apply default name
        ql = QLearner(env)
    else:
        ql = QLearner(env, model_name=model_file)

    if mode == "0": # Training
        policy = input("Select policy (0 = epsilon-greedy, 1 = random, 2 = combined): ").strip()
        
        if policy == "0": # Epsilon-Greedy policy
            rw_eps, eps_values = ql.DQN_Learning(1)
            np.save("./policy/reward_files", rw_eps)
            function_plot(rw_eps, eps_values)
        elif policy == "1": # Random policy
            rw_random, eps_values = ql.DQN_Learning(0)
            function_plot(rw_random, eps_values)
        elif policy == "2": # Both policies
            rw_random, eps_values = ql.DQN_Learning(0)
            rw_eps, eps_values = ql.DQN_Learning(1)
            np.save("./policy/reward_files", rw_eps)
            function_plot_combined(rw_eps, rw_random, eps_values)
        else:
            print("Policy not valid")
    elif mode == "1": # Running
        ql.run_policy()
    else:
        print("Input not valid")
