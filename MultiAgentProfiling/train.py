# Library Imports
import os
import shelve

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from MultiTD3 import Agent
import random
from gym.wrappers.time_limit import TimeLimit
from custom_pendulum import CustomPendulum
from gym.envs.classic_control.pendulum import PendulumEnv
import pickle

# Absolute Path
path = os.getcwd()

# Load the Environment
env = TimeLimit(env=CustomPendulum(PendulumEnv(), _seed=0), max_episode_steps=200)
use_predefined_actions_prob = 0.1
load_predefined_actions = True
preinject_episodes = 20
preinject_predefined_prob = 0.5
train_episodes = 300
mean_score_length = 100


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


# Init. Agents & replay buffer
agent1 = Agent(env, 'agent1')
agent2 = Agent(env, 'agent2')
agent3 = Agent(env, 'agent3')
Buffer = ReplayBuffer(3000000, env.observation_space.shape[0], env.action_space.shape[0])


# DEF. to transfer weights
def transfer_weights(best_agent):
    agent1.actor.set_weights(best_agent.actor.get_weights())
    agent2.actor.set_weights(best_agent.actor.get_weights())
    agent3.actor.set_weights(best_agent.actor.get_weights())

    agent1.target_actor.set_weights(best_agent.target_actor.get_weights())
    agent2.target_actor.set_weights(best_agent.target_actor.get_weights())
    agent3.target_actor.set_weights(best_agent.target_actor.get_weights())

    agent1.critic.set_weights(best_agent.critic.get_weights())
    agent2.critic.set_weights(best_agent.critic.get_weights())
    agent3.critic.set_weights(best_agent.critic.get_weights())

    agent1.target_critic.set_weights(best_agent.target_critic.get_weights())
    agent2.target_critic.set_weights(best_agent.target_critic.get_weights())
    agent3.target_critic.set_weights(best_agent.target_critic.get_weights())


global predefined_episodes


def train_agent(agent, i):
    score = 0
    # load init state
    use_predefined_actions = False
    if random.random() < use_predefined_actions_prob or i < preinject_episodes and random.random() < preinject_predefined_prob:
        print("using predefined episode")
        use_predefined_actions = True
        # randomly get predefined training data for this episode
        current_episode_inject_data = random.choice(predefined_episodes)

    starting_state = None
    if use_predefined_actions:
        starting_state = current_episode_inject_data["init_state"]
    obs = env.reset(starting_state=starting_state)

    done = False
    step_in_episode = 0
    while not done:
        # env.render()
        if use_predefined_actions:
            action = current_episode_inject_data["actions"][step_in_episode]
        else:
            action = agent.choose_action(obs)
        _obs, reward, done, info = env.step(action)
        Buffer.store_transition(obs, action, reward, _obs, done)
        obs = _obs
        score += reward
        step_in_episode += 1

    agent.learn(Buffer, 64)

    return score, use_predefined_actions


# Init. Training
agent1_scorelog = {}
agent2_scorelog = {}
agent3_scorelog = {}

agent1_mean_log = {}
agent2_mean_log = {}
agent3_mean_log = {}

agent1_best_mean_score = env.reward_range[0]
agent2_best_mean_score = env.reward_range[0]
agent3_best_mean_score = env.reward_range[0]

# load data to inject
if load_predefined_actions:
    with shelve.open("saved_actions") as f:
        predefined_episodes = f["data"]


def log_and_save(episode, agent_number, agent, score, used_predefined_actions, agent_score_log, agent_mean_log,
                 agent_best_mean_score):
    # add score to score log if it was a self determined action
    if not used_predefined_actions:
        agent_score_log[i] = score
    # if there is at least one score, calc the mean
    if len(agent_score_log.keys()) > 0:

        agent_mean_score = np.array(list(agent_score_log.values())[-mean_score_length:]).mean()
        agent_mean_log[i] = agent_mean_score
        print(
            f'Agent#{agent_number} -> Episode:{episode} \t ACC. Rewards: {score:4.2f} \t AVG. Rewards: {agent_mean_score:3.2f} \t Buffer Size:{Buffer.mem_cntr}')
        print("agent_score_log: " + str(len(agent_score_log)))
        with open(path + '/MultiAgentProfiling/data/agent' + str(agent_number) + '_scorelog', 'wb') as handle:
            pickle.dump(agent_score_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path + '/MultiAgentProfiling/data/agent' + str(agent_number) + '_meanlog', 'wb') as handle:
            pickle.dump(agent_mean_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save the best 'actor' model
        if agent_mean_score > agent_best_mean_score and not used_predefined_actions:
            agent_best_mean_score = agent_mean_score
            print(f"Saving 'agent{agent_number}' with best score:{agent_best_mean_score}")
            agent.actor.save_weights(path + '/MultiAgentProfiling/data/agent' + str(agent_number) + '.h5')
    return agent_best_mean_score


for i in range(train_episodes):
    print(f'\nTraining for Episode: {i}')

    # Agent#1 Training
    agent1_score, used_predefined_actions_1 = train_agent(agent1, i)
    agent1_best_mean_score = log_and_save(i, 1, agent1, agent1_score, used_predefined_actions_1, agent1_scorelog,
                                          agent1_mean_log, agent1_best_mean_score)
    # Agent#2 Training
    agent2_score, used_predefined_actions_2 = train_agent(agent2, i)
    log_and_save(i, 2, agent2, agent2_score, used_predefined_actions_2, agent2_scorelog, agent2_mean_log,
                 agent2_best_mean_score)

    # Agent#3 Training
    agent3_score, used_predefined_actions_3 = train_agent(agent3, i)
    log_and_save(i, 3, agent3, agent3_score, used_predefined_actions_3, agent3_scorelog, agent3_mean_log,
                 agent3_best_mean_score)

    # Init. transfer
    if i % 25 == 0:
        # Compute the best performing agent
        score_frame = np.array([agent1_best_mean_score, agent2_best_mean_score, agent3_best_mean_score])
        best_agent = np.argmax(score_frame)
        if best_agent == 0:
            best_agent = agent1
        elif best_agent == 1:
            best_agent = agent2
        elif best_agent == 2:
            best_agent = agent3
        print(f'Transfering Weights of {best_agent.name} to other agents...')
        transfer_weights(best_agent)
