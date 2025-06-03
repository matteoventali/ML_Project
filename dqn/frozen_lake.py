# Authors: Matteo Ventali and Valerio Spagnoli
# ML Project : DQN for Frozen Lake envinronment

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
from tensorflow.python.keras import layers
from tensorflow.python.keras import models

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, q_values):
        data = (state, q_values) # (s, <Q(s,a1),  ... , Q(s, aN)>)
        self.buffer.append(data)

    def sample(self, batch_size):
        batch = ran.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)

class DQN():
    def __init__(self, state_dim, num_actions):
        # Normalization layer
        self.normalizationLayer = layers.Normalization(input_shape=(state_dim,))
        
        # Creation of the neural network
        self.model = models.Sequential([
            self.normalizationLayer,
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_actions)  # output: Q(s,a) for all a
        ])
        self.model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))

    def train(self, batch):
        # Preparing the batch separating the state and the target values
        states = np.array([s for s, q in batch])   
        q_targets = np.array([q for s, q in batch])

        # Adapt of the state
        self.normalizationLayer.adapt(states)

        # Fit
        self.model.fit(states, q_targets, epochs=100)

    def get_qValue(self, state):
        # Computing the q_value for the state received
        state_input = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state_input, verbose=0)
        return q_values
    
class QLearner():
    def __init__(self, env:gym.Env, max_steps=5000000, gamma=0.99, alpha=0.1, end_eps=0.01, start_eps=1.0,  eps_decay=0.9999):
        self.env = env
        self.max_steps = max_steps        
        self.gamma = gamma
        self.alpha = alpha
        self.end_eps = end_eps
        self.eps = start_eps
        self.eps_decay = eps_decay
        
        
    def _espilon_update(self):
        self.eps = max(self.eps_decay * self.eps, self.end_eps)

    
    def _next_action(self, current_state):
        n = ran.random()
        if n < self.eps: # Exploration
            return ran.randint(0,3)
        else: # Exploitation
            return np.argmax(self.q_table[current_state])

              
    def DQN_Learning(self):
        # Creation of the NN representing the Q-table

        # Creation of the dataset implementing the replay memory

        # Starting of the environment
        s, _ = self.env.reset()

        for n_step in range(self.max_steps):
            print(f"step n: {n_step}")
            # Select the action to be executed
            a = self._next_action(s)

            # Execution of a
            ns, reward, terminated, _, _ = self.env.step(a)
            total_reward += reward
            
            # Update Q_table
            # Q(s,a) = (1 - alpha)*Q(s,a) + alpha*[r + y max_a'{Q(s',a')}]
            # Q(s,a) = Q(s,a) + alpha[r + y*max_a'{Q(s',a')} - Q(s,a)]

            # Updating new state
            if terminated:
                print(f"episode end, total reward: {total_reward}")
                self._espilon_update()
                s, _ = self.env.reset()
            else:
                s = ns
        


if __name__ == "__main__":
    # Lunar Lander Environment
    #env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    #env = gym.make('FrozenLake-v1', desc=["FFFH", "HGHF", "FFFF", "SFHF"], map_name=None, is_slippery=False)
    #env = gym.make('FrozenLake-v1', render_mode="human", desc=generate_random_map(size=5), is_slippery=True)
    pass
    