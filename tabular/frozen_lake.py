# Authors: Matteo Ventali and Valerio Spagnoli
# ML Project : tabular Q-learning for Frozen Lake envinronment

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from collections import defaultdict
import random as ran
import pickle
from pprint import pprint
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

class QLearner():
    def __init__(self, env:gym.Env, max_steps=5000000, gamma=0.99, alpha=0.1, end_eps=0.01, start_eps=1.0,  eps_decay=0.9999):
        self.env = env
        self.max_steps = max_steps        
        self.gamma = gamma
        self.alpha = alpha
        self.end_eps = end_eps
        self.eps = start_eps
        self.eps_decay = eps_decay
        self.policy_name = 'policy_frozen_lake'
        
    
    def _espilon_update(self):
        self.eps = max(self.eps_decay * self.eps, self.end_eps)

    
    def _next_action(self, current_state):
        n = ran.random()
        if n < self.eps: # Exploration
            return ran.randint(0,self.env.action_space.n - 1)
        else: # Exploitation
            return np.argmax(self.q_table[current_state])

              
    def tabular_QLearning(self):
        # Q-table creation dynamically. It is implemented as a dictionary in which:
        # - the keys are the states (observation collected)
        # - the values are array of 4 components (number of actions available), representing Q(s,a)
        # - when a state is not already in the dictionary, then it is added with a value of (0,0,0,0). This happens the first time
        #   the agent ends up in that state
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))

        # Starting of the environment
        s, _ = self.env.reset()

        total_reward = 0
        
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
            self.q_table[s][a] += self.alpha * (reward + self.gamma * max(self.q_table[ns]) - self.q_table[s][a])

            # Updating new state
            if terminated:
                print(f"episode end, total reward: {total_reward}")
                self._espilon_update()
                s, _ = self.env.reset()
            else:
                s = ns
        
        # Dump q_table
        with open(self.policy_name, "wb") as f:
            pickle.dump(dict(ql.q_table), f)

    
    def load_policy(self):
        with open(self.policy_name, "rb") as f:
            self.policy = defaultdict(lambda: np.zeros(self.env.action_space.n),pickle.load(f))
        
    
    def run_policy(self):
        self.load_policy()

        n_success = 0
        
        for i in range(0,1000):
            s, _ = self.env.reset()
            
            done = False
            rw = 0

            while not done:
                a = np.argmax(self.policy[s])
                ns, r, done, _, _ = self.env.step(a)
                rw += r
                s = ns

                if s == self.env.observation_space.n - 1:
                    n_success += 1
            
            print(f"Episode {i}: final state = {s}, total reward = {rw:.2f}")
            print(f"Reward obtained: {n_success}")
        

if __name__ == "__main__":
    # Lunar Lander Environment
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    #env = gym.make('FrozenLake-v1', desc=["FFFH", "HGHF", "FFFF", "SFHF"], map_name=None, is_slippery=False)
    #env = gym.make('FrozenLake-v1', render_mode="human", desc=generate_random_map(size=5), is_slippery=True)

    ql = QLearner(env)
    #ql.tabular_QLearning()
    ql.run_policy()