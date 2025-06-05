# Authors: Matteo Ventali and Valerio Spagnoli
# ML Project : DQN for Lunar Lander envinronment

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import random as ran
from gymnasium.spaces import Box
from collections import deque
from pprint import pprint
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense, Normalization #type: ignore


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
    def __init__(self, state_dim, num_actions):
        # Creation of the neural network
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(8,)),
            Dense(64, activation='relu'),
            Dense(num_actions)  # output: Q(s,a) for all a
        ])
        self.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))

    def train(self, batch):
        # Preparing the batch separating the state and the target values
        states = np.array([s for s, q in batch])   
        q_targets = np.array([q for s, q in batch])

        # Fit
        self.model.fit(states, q_targets)

    def predict_qValue(self, state):
        # Computing the q_value for the state received
        state_input = np.expand_dims(state, axis=0)
        q_values = self.model.predict_on_batch(state_input)
        return q_values

    
class QLearner():
    def __init__(self, env:gym.Env, max_episodes=3000, gamma=0.9, alpha=0.1, end_eps=0.01, start_eps=1.0,  eps_decay=0.999, model_name="dqn_model.keras"):
        self.env = env
        self.max_episodes = max_episodes        
        self.gamma = gamma
        self.alpha = alpha
        self.end_eps = end_eps
        self.eps = start_eps
        self.eps_decay = eps_decay
        self.batch_dimension = 64
        self.model_name = "./dqn_models/" + model_name
        
    def _espilon_update(self):
        self.eps = max(self.eps_decay * self.eps, self.end_eps)

    def _next_action(self, current_state, q_network : DQN):
        n = ran.random()
        if n < self.eps: # Exploration
            return self.env.action_space.sample()
        else: # Exploitation
            q_values = q_network.predict_qValue(current_state)[0]
            a = np.argmax(q_values)
            return a

    def _prepareBatch(self, batch, q_network : DQN):
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
            row[1][action] = q_values_s[action] + self.alpha * (t[2] + self.gamma * np.max(q_values_ns) * (1 - int(t[4])) - q_values_s[action])
            
            training_set.append(row)

        return training_set

    def _normalize(self, state):
        amplitudes = [ 2.5, 2.5, 10., 10., 6.2831855, 10., 1., 1.]
        result = state/amplitudes 
        return result

    def DQN_Learning(self):
        # Creation of the NN representing the Q-table
        q_network = DQN(8, self.env.action_space.n)

        # Creation of the dataset implementing the replay memory
        memory = ReplayBuffer(10000)

        # Starting of the environment
        s, _ = self.env.reset()

        # Array for collecting total rewards
        total_rewards = []

        for n_episode in range(self.max_episodes):
            print(f"Episode n: {n_episode}")
            episode_reward = 0
            truncated = terminated = False
            while not (truncated or terminated):
                # Select the action to be executed
                a = self._next_action(s, q_network)

                # Execution of a
                ns, reward, terminated, truncated, _ = self.env.step(a)
                episode_reward += reward
                done = terminated or truncated
                
                # (s,a,r,s') in replay buffer
                memory.add(self._normalize(s), a, reward, self._normalize(ns), done)

                # get a sample batch for training
                if ( len(memory) > self.batch_dimension ):
                    batch = memory.sample(self.batch_dimension)
                    # Preparing the batch
                    training_set = self._prepareBatch(batch, q_network)
                    q_network.train(training_set)

                # Updating new state
                if terminated:
                    self._espilon_update()
                    s, _ = self.env.reset()
                else:
                    s = ns
            
            # Stats of the episode
            print(f"(episode {n_episode} {episode_reward})")
            self._espilon_update()
            s, _ = self.env.reset()
            total_rewards.append(episode_reward)

        self._save_policy(q_network)
        return total_rewards

    def _save_policy(self, q_network : DQN):
        # Policy saving. Weigths of the neural network
        q_network.model.save(self.model_name)

    def _load_policy(self, q_network : DQN):
        # Policy loading
        q_network.model.load(self.model_name)

    def run_policy(self):
        # NN for Q-Values already trained
        q_network = DQN(8, self.env.action_space.n)
        self._load_policy(q_network)
        
        total_reward = 0
        episodes_reward = []
        n_episodes = 1000

        for i in range(0,n_episodes):
            s, _ = self.env.reset()
            
            terminated = truncated = False
            rw = 0

            while not (terminated or truncated):
                # Retrieve the action with maximum q_value
                q_values = q_network.predict_qValue(s)[0]
                a = np.argmax(q_values)

                ns, r, terminated, truncated, _ = self.env.step(a)
                rw += r
                s = ns

            total_reward += rw
            episodes_reward.append(rw)
            print(f"Episode {i}: final state = {s}, total reward = {rw:.2f}")
        
        print(f"Mean Reward: {total_reward/n_episodes}")
        print(f"Mean Episode Reward: {np.mean(episodes_reward)}")



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Lunar Lander Environment
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    
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
            rw_eps= ql.DQN_Learning()
            np.save("./policy/reward_files_dqn", rw_eps)
            plt.plot(np.convolve(rw_eps, np.ones(1000)/1000, mode="valid"), label='Epsilon Greedy policy 1000', color="red")
            plt.show()
        elif policy == "1": # Random policy
            rw_random = ql.DQN_Learning(1)
            plt.plot(np.convolve(rw_random, np.ones(2000)/2000), label='Random policy')
            plt.show()
        elif policy == "2": # Both policies
            rw_random = ql.DQN_Learning(1)
            rw_eps = ql.DQN_Learning()
            np.save("./policy/reward_files_dqn", rw_eps)
            plt.plot(np.convolve(rw_random, np.ones(2000)/2000), label='Random policy', color="green")
            plt.plot(np.convolve(rw_eps, np.ones(1000)/1000, mode="valid"), label='Epsilon Greedy policy 1000', color="red")
            plt.show()
        else:
            print("Policy not valid")
    elif mode == "1": # Running
        ql.run_policy()
    else:
        print("Input not valid")
