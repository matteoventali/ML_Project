# Authors: Matteo Ventali and Valerio Spagnoli
# ML Project : tabular Q-learning for Lunar Lander envinronment

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from collections import defaultdict
import random as ran
from pprint import pprint
import pickle

class ObservationDiscretizer:
    def __init__(self, space: Box, bins_per_var=6):
        assert space.shape == (8, ), "Space box error" # The observation is a 1D array of 8 elements
        self.bins = bins_per_var
        self.discrete_bins = []
        for i in range(6):
            self.discrete_bins.append(np.linspace(space.low[i], space.high[i], bins_per_var+1)[1:-1])

    def discretize(self, obs):
        result = []
        for i in range(6):
            result.append(np.digitize(obs[i], self.discrete_bins[i]))
        
        # No need to discretize boolean variables
        result.append(int(obs[6]))
        result.append(int(obs[7]))
        return tuple(result)
    

class QLearner():
    def __init__(self, env:gym.Env, max_steps=1000000, gamma=0.9, alpha=0.3, end_eps=0.01, start_eps=1.0,  eps_decay=0.999):
        self.env = env
        self.max_steps = max_steps        
        self.gamma = gamma
        self.alpha = alpha
        self.end_eps = end_eps
        self.eps = start_eps
        self.eps_decay = eps_decay
        self.discretizer = ObservationDiscretizer(self.env.observation_space)
        self.policy_name = 'policy_lunar_lander'
        
    
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

        success = 0
        total_reward = 0

        # Starting of the environment
        s, _ = self.env.reset()
        s = self.discretizer.discretize(s)

        for n_step in range(self.max_steps):
            print(f"step n: {n_step}")
            # Select the action to be executed
            a = self._next_action(s)

            # Execution of a
            ns, reward, terminated, _, _ = self.env.step(a)
            total_reward += reward
            ns = self.discretizer.discretize(ns)

            # Update Q_table
            # Q(s,a) = (1 - alpha)*Q(s,a) + alpha*[r + y max_a'{Q(s',a')}]
            # Q(s,a) = Q(s,a) + alpha[r + y*max_a'{Q(s',a')} - Q(s,a)]
            self.q_table[s][a] += self.alpha * (reward + self.gamma * max(self.q_table[ns]) - self.q_table[s][a])

            # Updating new state
            if terminated:
                print(f"episode end, total reward: {total_reward}")
                self._espilon_update()
                s, _ = self.env.reset()
                s = self.discretizer.discretize(s)
                if total_reward > 200:
                    success += 1
                total_reward = 0
            else:
                s = ns
        
        # Dump q_table
        with open(self.policy_name, "wb") as f:
            pickle.dump(dict(ql.q_table), f)

        return success
    
    
    def load_policy(self):
        with open(self.policy_name, "rb") as f:
            self.policy = defaultdict(lambda: np.zeros(self.env.action_space.n),pickle.load(f))
        
    
    def run_policy(self):
        self.load_policy()

        n_success = 0
        
        for i in range(0,1000):
            s, _ = self.env.reset()
            s = self.discretizer.discretize(s)
            
            done = False
            rw = 0

            while not done:
                a = np.argmax(self.policy[s])
                ns, r, done, _, _ = self.env.step(a)
                ns = self.discretizer.discretize(ns)
                rw += r
                s = ns

            if rw >= 200:
                n_success += 1
            
            print(f"Episode {i}: final state = {s}, total reward = {rw:.2f}")
            print(f"Reward obtained: {n_success}")
        

if __name__ == "__main__":
    # Lunar Lander Environment
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    
    ql = QLearner(env)
    ql.tabular_QLearning()
    ql.run_policy()