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
    def __init__(self, env:gym.Env, max_episodes=30000, gamma=0.9, alpha=0.1, end_eps=0.01, start_eps=1.0,  eps_decay=0.999):
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
        total_rewards = []
        
        # Training of episodes
        for n_episode in range(self.max_episodes):
            
            terminated = truncated = False
            episode_reward = 0
            
            while not (terminated or truncated): # Single episode
                # Select the action to be executed
                a = self._next_action(s, modality)

                # Execution of a
                ns, reward, terminated, truncated, _ = self.env.step(a)
                episode_reward += reward
                ns = discretize(ns)

                # Update Q_table
                # Q(s,a) = (1 - alpha)*Q(s,a) + alpha*[r + y max_a'{Q(s',a')}]
                # Q(s,a) = Q(s,a) + alpha[r + y*max_a'{Q(s',a')} - Q(s,a)]
                self.q_table[s][a] += self.alpha * (reward + self.gamma * max(self.q_table[ns]) - self.q_table[s][a])

                # Updating new state
                if not (terminated or truncated):
                    s = ns
            
            # Stats of the episode
            print(f"(episode {n_episode} {episode_reward})")
            self._espilon_update()
            s, _ = self.env.reset()
            s = discretize(s)
            total_rewards.append(episode_reward)

        # Dump q_table
        with open(self.policy_name, "wb") as f:
            pickle.dump(dict(ql.q_table), f)

        return total_rewards
            

    def load_policy(self):
        with open(self.policy_name, "rb") as f:
            self.policy = defaultdict(lambda: np.zeros(self.env.action_space.n),pickle.load(f))
        
    
    def run_policy(self):
        self.load_policy()
        
        total_reward = 0
        episodes_reward = []
        n_episodes = 1000

        for i in range(0,n_episodes):
            s, _ = self.env.reset()
            s = discretize(s)
            
            terminated = truncated = False
            rw = 0

            while not (terminated or truncated):
                a = np.argmax(self.policy[s])
                ns, r, terminated, truncated, _ = self.env.step(a)
                ns = discretize(ns)
                rw += r
                s = ns

            total_reward += rw
            episodes_reward.append(rw)
            print(f"Episode {i}: final state = {s}, total reward = {rw:.2f}")
        
        print(f"Mean Reward: {total_reward/n_episodes}")
        print(f"Mean Episode Reward: {np.mean(episodes_reward)}")
        

if __name__ == "__main__":
    # Lunar Lander Environment
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    
    # Training
    ql = QLearner(env)
    #rw_random = ql.tabular_QLearning(0)
    #print("\nRestarting training")
    #rw_eps = ql.tabular_QLearning()
    #np.save("./tabular/policies/reward_files", rw_eps)
    
    # Results plot
    #plt.plot(np.convolve(rw_random, np.ones(2000)/2000), label='Random policy')
    #plt.plot(np.convolve(rw_eps, np.ones(50)/50), label='Epsilon Greedy policy 50')
    #plt.plot(np.convolve(rw_eps, np.ones(500)/500), label='Epsilon Greedy policy 500')
    #plt.plot(np.convolve(rw_eps, np.ones(1000)/1000, mode="valid"), label='Epsilon Greedy policy 1000')
    #plt.plot(np.convolve(rw_eps, np.ones(2000)/2000, mode="valid"), label='Epsilon Greedy policy 2000')
    #plt.plot(np.convolve(rw_eps, np.ones(2500)/2500, mode="valid"), label='Epsilon Greedy policy 2500')
    #plt.show()

    ql.run_policy()