# Library Import
import numpy as np
import matplotlib.pyplot as plt
import os
np.set_printoptions(precision=4)

# Import Data
#agent1 = np.load(os.getcwd()+'/MultiAgentProfiling/data/agent1_meanlog.npy')
#agent2 = np.load(os.getcwd()+'/MultiAgentProfiling/data/agent2_avglog.npy')
#agent3 = np.load(os.getcwd()+'/MultiAgentProfiling/data/agent3_avglog.npy')
#agent0 = np.load(os.getcwd()+'/SingleAgentProfiling/data/avg_history.npy')

#Plot and Save the Graphs
#plt.figure(1)
#plt.plot(agent0, color='black', label='solo_agent')
#plt.plot(agent1, color='red', label='inteam_agent1')
#plt.plot(agent2, color='blue', label='inteam_agent2')
#plt.plot(agent3, color='green', label='inteam_agent3')
#plt.grid(True)
#plt.xlabel('Episodes')
#plt.ylabel('Avg. Rewards')
#plt.title("Collective Training Profile")
#plt.legend(loc='best')
#plt.savefig(os.getcwd()+'/data/Collective Training Profile.png')

# Import Data
solo_score = np.load(os.getcwd()+'/data/solo_score.npy')
team_score = np.load(os.getcwd()+'/data/team_score.npy')
solo_eps = np.load(os.getcwd()+'/data/solo_eps.npy')
team_eps = np.load(os.getcwd()+'/data/team_eps.npy')

#Plot and Save the Graphs
plt.figure(2)
plt.plot(solo_score, label='Solo Agent')
plt.plot(team_score, label='Agents in Team')
plt.grid(True)
plt.xlabel('Episodes')
plt.ylabel('Total Rewards')
plt.title("Agent Testing Profile")
plt.legend(loc='best')
plt.savefig(os.getcwd()+'/data/Agent Testing Profile.png')

print('Analysis...')
print(f'Single Agent Sum.Score: {solo_score[-1]} \t Multi Agent Sum.Score: {team_score[-1]}')
print(f'Single Agent Mean Score: {solo_score[-1]/len(solo_score)} \t Multi Agent Mean Score: {team_score[-1]/len(team_score)}')
print(f'Scores in Ratio :\nSingle Agent : Multi Agent ={1/(solo_score[-1]/team_score[-1])}\nMulti Agent : Single Agent ={1/(team_score[-1]/solo_score[-1])}')

