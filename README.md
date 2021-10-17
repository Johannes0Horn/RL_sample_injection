# Sample Injection for Reinforcement Learning evaluation

This repository aims to analyse if the method called "Sample Injection for Reinforcement Learning" proposed in the paper "[Towards generating complex programs represented as node-trees with reinforcement learning](https://github.com/Johannes0Horn/RL_sample_injection/blob/main/Towards%20generating%20complex%20programs%20represented%20as%20node-trees%20with%20reinforcement%20learning.pdf)", in general can be helpful for different reinforcement learning algorithms and evironments to achieve better and/or faster results.

# Done:

## Test sample injection on a state of the art reinforcement algorithm in an easy environment:

To do so, the best performing algorithm([TD3](https://www.researchgate.net/profile/Wenfeng-Zheng/publication/341648433_Twin-Delayed_DDPG_A_Deep_Reinforcement_Learning_Technique_to_Model_a_Continuous_Movement_of_an_Intelligent_Robot_Agent/links/5ed9ae3e92851c9c5e816d19/Twin-Delayed-DDPG-A-Deep-Reinforcement-Learning-Technique-to-Model-a-Continuous-Movement-of-an-Intelligent-Robot-Agent.pdf)) according to the [OpenAI gym Leaderboard](https://github.com/openai/gym/wiki/Leaderboard) for the OpenAI gym Environment ["Pendulum-v0"](https://gym.openai.com/envs/Pendulum-v0/) implemented by [Kanishk Navale](https://github.com/KanishkNavale) in a Multi Agent-manner, was [bugfixed](https://github.com/KanishkNavale/Cooperative-Deep-RL-Multi-Agents/issues/3) and modified: [Cooperative-Deep-RL-Multi-Agents](https://github.com/KanishkNavale/Cooperative-Deep-RL-Multi-Agents).

To evaluate sample injection vs. the standard learning approach, 3 experiments were run.

### Preparation: 

Train a model from scratch with 1000 Episodes and create samples for almost perfect Episode execution:
- in MultiAgentProfiling/train.py set:
    - use_predefined_actions_prob = 0
    - load_predefined_actions = False
    - preinject_episodes = 0
    - n_games = 1000
- run `python MultiAgentProfiling/train` to train the model.
- Create samples from the trained model:
    - in test_agents.py set:
        - save_actions = True
        - use_predefined_actions = False
    - run `python test_agents.py` to in infere the model and save action samples.
    - (optional: verify good performance of predefined actions):
        -  in test_agents.py set:
            -  use_predefined_actions = True
            -  save_actions = False
            -  uncomment line 91: `# env.render()` to view the actions
        - run `python test_agents.py` again.

### 1. Train the model for 300 Episodes without sample injection from scratch:
    - in MultiAgentProfiling/train.py set:
        - use_predefined_actions_prob = 0
        - load_predefined_actions = False
        - preinject_episodes = 0
    - run `python MultiAgentProfiling/train.py` to train the model.
    - Create a new Folder in MultiAgentProfiling/data/ called *your experiment name1* 
    - Move the the created train logs from MultiAgentProfiling/data/ to MultiAgentProfiling/data/*your experiment name1* 

### 2. Train the model for 300 Episodes with a steady probability of 10% per Episode to be sample injected
    - in MultiAgentProfiling/train.py set:
        - use_predefined_actions_prob = 0.1
        - load_predefined_actions = True
        - preinject_episodes = 0
    - run `python MultiAgentProfiling/train.py` to train the model.
    - Create a new Folder in MultiAgentProfiling/data/ called *your experiment name2* 
    - Move the the created train logs from MultiAgentProfiling/data/ to MultiAgentProfiling/data/*your experiment name2* 

### 3. Train the model for 20 Episodes with a steady probability of 50% per Episode to be sample injected. The reduce the probability of sample injecion to 10% until 300 total        Episodes of training are reached.
    - in MultiAgentProfiling/train.py set:
        - use_predefined_actions_prob = 0
        - load_predefined_actions = False
        - preinject_episodes = 0
        - preinject_episodes = 20
        - preinject_predefined_prob = 0.5
    - run `python MultiAgentProfiling/train.py` to train the model.
    - Create a new Folder in MultiAgentProfiling/data/ called *your experiment name3* 
    - Move the the created train logs from MultiAgentProfiling/data/ to MultiAgentProfiling/data/*your experiment name3* 

### Results:
In MultiAgentProfiling/profile set:
- experiment1 = *your experiment name1*
- experiment2 = *your experiment name2*
- experiment3 = *your experiment name3*

run `python MultiAgentProfiling/profile` to create the plot.


|Experiments Training Profile of three Agents results averaged | 
|:--:|
|<img src="/MultiAgentProfiling/data/agents_merged Training Profile.png" width="800">|

While all models in the end achieve the same training performance, one can tell that the models trained with sample injection converge faster in the beginning until ~ Episode 110. This looks promising for environments which are harder to solve with a bigger action and observation space, so that algorithms which are "sample injected" might find high rewarding actions faster. 


# TODO:
- [ ] Benchmark sample injection vs supervised pre-training.
- [ ] Test sample injection on a state of the art reinforcement algorithm in a more difficult environment like [Ant-v2](https://gym.openai.com/envs/Ant-v2/)
