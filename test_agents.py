# Library Imports
import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import numpy as np
from MultiTD3 import Agent
import shelve
from gym.wrappers.time_limit import TimeLimit
from custom_pendulum import CustomPendulum
from gym.envs.classic_control.pendulum import PendulumEnv

save_actions = False
use_predefined_actions = False

if use_predefined_actions:
    global predefined_episodes
    with shelve.open("saved_actions") as f:
        predefined_episodes = f["data"]

        # Absolute Path
path = os.getcwd()

# Load the Environment
env = TimeLimit(env=CustomPendulum(PendulumEnv(), _seed=1), max_episode_steps=200)

# Init. Agent
team_agent1 = Agent(env, 'team_agent1')
solo_agent = Agent(env, 'solo_agent')

# Load the Trained Actor
for i in range(1):
    done = False
    obs = env.reset()
    while not done:
        action = team_agent1.choose_action(obs)
        action = solo_agent.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        obs = _obs

team_agent1.actor.load_weights(path + '/MultiAgentProfiling/data/agent1.h5')
solo_agent.actor.load_weights(path + '/SingleAgentProfiling/data/agent.h5')

# Init. & Profile the Solo and Team Agent
print(f'Testing Solo Agent...')
solo_scorelog = []
solo_epslog = []
score = 0
env = TimeLimit(env=CustomPendulum(PendulumEnv(), _seed=1), max_episode_steps=200)
for i in range(100):
    done = False
    obs = env.reset()

    while not done:
        # env.render()

        action = solo_agent.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        obs = _obs
        score += reward
    solo_scorelog.append(score)
    solo_epslog.append(i)
env.close()
np.save(path + '/data/solo_score', solo_scorelog, allow_pickle=False)
np.save(path + '/data/solo_eps', solo_epslog, allow_pickle=False)

print(f'Testing Team Agent...')
team_scorelog = []
team_epslog = []
score = 0
episodes = []
env = TimeLimit(env=CustomPendulum(PendulumEnv(), _seed=1), max_episode_steps=200)
for i in range(100):

    done = False

    # load init state
    starting_state = None
    if use_predefined_actions:
        starting_state = predefined_episodes[i]["init_state"]
    obs = env.reset(starting_state=starting_state)

    # save variables
    curr_episode = {}
    curr_episode_actions = []

    # save init state
    # convert obs to state
    curr_episode["init_state"] = env.get_state()
    step_in_episode = 0
    while not done:
        # env.render()
        # Taking a step using agent1
        action1 = solo_agent.choose_action(obs)
        if use_predefined_actions:
            action1 = predefined_episodes[i]["actions"][step_in_episode]
        curr_episode_actions.append(action1)

        agent1_obs, agent1_reward, done, info = env.step(action1)
        obs = agent1_obs

        step_in_episode += 1
        score += agent1_reward

    # append actions to episodes object

    curr_episode["actions"] = curr_episode_actions
    episodes.append(curr_episode)

    team_scorelog.append(score)
    team_epslog.append(i)

# pickle initial state and all actions per episode
if save_actions:
    with shelve.open("saved_actions") as f:
        f["data"] = episodes

env.close()
np.save(path + '/data/team_score', team_scorelog, allow_pickle=False)
np.save(path + '/data/team_eps', team_epslog, allow_pickle=False)
