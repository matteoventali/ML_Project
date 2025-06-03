# Authors: Matteo Ventali and Valerio Spagnoli
# ML Project : tabular Q-learning for Lunar Lander envinronment

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from collections import defaultdict
import random as ran
from pprint import pprint
import pickle

def discretize(obs):
        result = []
        
        # Interval array
        x_intervals     = [-0.5, 0.5]
        y_intervals     = [-0.1, 0.1, 1.5]
        vx_intervals    = [-7.5, -5, -0.3, -0.1, 0.1, 0.3, 5, 7.5]
        vy_intervals    = [-7.5, -5, -0.3, -0.1, 0.1, 0.3, 5, 7.5]
        theta_intervals = [-1.25663706,  -0.1, 0.1, 1.25663706]
        omega_intervals = [-7.5, -5, -0.1, 0.1, 5, 7.5]
        
        result.append(np.digitize(obs[0], x_intervals))
        result.append(np.digitize(obs[1], y_intervals))
        result.append(np.digitize(obs[2], vx_intervals))
        result.append(np.digitize(obs[3], vy_intervals))
        result.append(np.digitize(obs[4], theta_intervals))
        result.append(np.digitize(obs[5], omega_intervals))

        # No need to discretize boolean variables
        result.append(int(obs[6]))
        result.append(int(obs[7]))
        return tuple(result)
    

class QLearner():
    def __init__(self, env:gym.Env, max_episodes=10000, gamma=0.9, alpha=0.3, end_eps=0.01, start_eps=1.0,  eps_decay=0.999):
        self.env = env
        self.max_episodes = max_episodes        
        self.gamma = gamma
        self.alpha = alpha
        self.end_eps = end_eps
        self.eps = start_eps
        self.eps_decay = eps_decay
        self.policy_name = 'policy_lunar_lander'
        
    
    def _espilon_update(self):
        self.eps = max(self.eps_decay * self.eps, self.end_eps)

    
    def _next_action(self, current_state, modality):
        n = ran.random()
        if modality == 0  or n < self.eps: # Exploration
            return ran.randint(0,self.env.action_space.n - 1)
        else: # Exploitation
            return np.argmax(self.q_table[current_state])

              
    def tabular_QLearning(self, modality=1): # 0 for random, 1 for eps-greedy policy
        # Q-table creation dynamically. It is implemented as a dictionary in which:
        # - the keys are the states (observation collected)
        # - the values are array of 4 components (number of actions available), representing Q(s,a)
        # - when a state is not already in the dictionary, then it is added with a value of (0,0,0,0). This happens the first time
        #   the agent ends up in that state
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))

        # Starting of the environment
        s, _ = self.env.reset()
        s = discretize(s)

        # Array for collecting total rewards
        total_rewards = [0] * self.max_episodes
        
        # Training of episodes
        for n_episode in range(self.max_episodes):
            terminated = False

            while not terminated: # Single episode
                # Select the action to be executed
                a = self._next_action(s, modality)

                # Execution of a
                ns, reward, terminated, truncated, _ = self.env.step(a)
                total_rewards[n_episode] += reward
                ns = discretize(ns)

                # Update Q_table
                # Q(s,a) = (1 - alpha)*Q(s,a) + alpha*[r + y max_a'{Q(s',a')}]
                # Q(s,a) = Q(s,a) + alpha[r + y*max_a'{Q(s',a')} - Q(s,a)]
                self.q_table[s][a] += self.alpha * (reward + self.gamma * max(self.q_table[ns]) - self.q_table[s][a])

                # Updating new state
                if not terminated:
                    s = ns
            
            # Stats of the episode
            print(f"(episode {n_episode} {total_rewards[n_episode]})")
            self._espilon_update()
            s, _ = self.env.reset()
            s = discretize(s)

        # Dump q_table
        with open(self.policy_name, "wb") as f:
            pickle.dump(dict(ql.q_table), f)

        return total_rewards[-50:]
            

    def load_policy(self):
        with open(self.policy_name, "rb") as f:
            self.policy = defaultdict(lambda: np.zeros(self.env.action_space.n),pickle.load(f))
        
    
    def run_policy(self):
        self.load_policy()

        n_success = 0
        
        for i in range(0,1000):
            s, _ = self.env.reset()
            s = discretize(s)
            
            done = False
            rw = 0

            while not done:
                a = np.argmax(self.policy[s])
                ns, r, done, _, _ = self.env.step(a)
                ns = discretize(ns)
                rw += r
                s = ns

            if rw >= 200:
                n_success += 1
            
            print(f"Episode {i}: final state = {s}, total reward = {rw:.2f}")
            print(f"Reward obtained: {n_success}")
        

if __name__ == "__main__":
    # Lunar Lander Environment
    env = gym.make("LunarLander-v3", render_mode="human", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    
    # Training
    ql = QLearner(env)
    #rw_random = ql.tabular_QLearning(0)
    #print("\nRestarting training")
    #rw_eps = ql.tabular_QLearning()
    
    # Results plot
    #plt.plot(rw_random, label='Random policy')
    #plt.plot(rw_eps, label='Epsilon Greedy policy')
    #plt.show()

    ql.run_policy()

    #print(discretize((-5.1, 0, 6.3, 7.1, 2.544, 4.1, 0, 0)))