# Library Imports
import os
import numpy as np
import matplotlib.pyplot as plt

# Import Data
avg = np.load(os.getcwd()+'/SingleAgentProfiling/data/avg_history.npy')
sum = np.load(os.getcwd()+'/SingleAgentProfiling/data/score_history.npy')

#Plot and Save the Graphs
plt.plot(sum, color='red', alpha=0.5, label='Summed Rewards')
plt.plot(avg, color='black', label='Average Rewards')
plt.grid(True)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Single Agent Training Profile')
plt.legend(loc='best')
plt.savefig(os.getcwd()+'/SingleAgentProfiling/data/Training_Profile.png')