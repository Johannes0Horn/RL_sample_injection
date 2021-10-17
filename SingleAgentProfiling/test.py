# Library Imports
import os
import numpy as np
import gym
from TD3 import Agent

# Absolute Path
path = os.getcwd()

# Load the Environment
env = gym.make('Pendulum-v0')

# Init. Agent
agent = Agent(env)

# Load the Trained Actor
for i in range(10):
    done = False
    obs = env.reset()
    while not done:
        action = agent.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        agent.store_exp(obs, action, reward, _obs, done)
        obs = _obs
        
agent.actor.load_weights(path+'/SingleAgentProfiling/data/actor.h5')

# Init. Training
for i in range(10):
    done = False
    
    # Initial Reset of Environment
    obs = env.reset()
    while not done:
        env.render()
        action = agent.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        obs = _obs

# Close the Env.
env.close()