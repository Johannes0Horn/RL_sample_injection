# Library Imports
import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import gym
from MultiTD3 import Agent
import random
from gym.wrappers.time_limit import TimeLimit
from custom_pendulum import CustomPendulum
from gym.envs.classic_control.pendulum import PendulumEnv

# Absolute Path
path = os.getcwd()

# Load the Environment

env = TimeLimit(env=CustomPendulum(PendulumEnv(), _seed=0), max_episode_steps=200)

# Init. Training
n_games = 300
score_history = []
avg_history = []
best_score = env.reward_range[0]
avg_score = 0


# Init. Global Replay Buffer
class ReplayBuffer:
    """Defines the Buffer dataset from which the agent learns"""

    def __init__(self, max_size, input_shape, dim_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, dim_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        _states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, _states, dones


# Init. Agent & replay buffer
agent = Agent(env, 'agent')

Buffer = ReplayBuffer(1000000, env.observation_space.shape[0], env.action_space.shape[0])

for i in range(n_games):
    score = 0
    done = False

    # Initial Reset of Environment
    obs = env.reset()

    while not done:
        action = agent.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        Buffer.store_transition(obs, action, reward, _obs, done)
        obs = _obs
        score += reward

    # Optimize the Agent    
    agent.learn(Buffer, 64)

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    avg_history.append(avg_score)

    if avg_score > best_score:
        best_score = avg_score
        print(f"Saving 'agent' with best score:{best_score}")
        agent.actor.save_weights(path + '/SingleAgentProfiling/data/agent.h5')
        print(f'Episode:{i} \t ACC. Rewards: {score:3.2f} \t AVG. Rewards: {avg_score:3.2f} \t *** MODEL SAVED! ***')
    else:
        print(f'Episode:{i} \t ACC. Rewards: {score:3.2f} \t AVG. Rewards: {avg_score:3.2f}')

    # Save the Training data and Model Loss
    np.save(path + '/SingleAgentProfiling/data/score_history', score_history, allow_pickle=False)
    np.save(path + '/SingleAgentProfiling/data/avg_history', avg_history, allow_pickle=False)
