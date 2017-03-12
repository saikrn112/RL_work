

import sys
import math
import rlglue.RLGlue as RLGlue
import matplotlib.pyplot as plt
import numpy as np

def offlineDemo():
	results=[];
	avgsteps= [];
	avgreward=[];
	episodeResult=Startexperiment();
	printScore(0,0,episodeResult);
	results.append(episodeResult);
	episodes = 10
	n = np.arange(1,episodes+2,1);

	for j in range(0,2): # 50 runs 
		RLGlue.RL_cleanup();
		for i in range(0,episodes): # Let's try 100 episodes for learning 
			RLGlue.RL_env_message("set-random-start-state");
			episodeResult=Startexperiment();
			printScore((j+1),(i+1),episodeResult);
			results.append(episodeResult);
		avgsteps.append([x[1] for  x in results])
		avgreward.append([x[0] for  x in results])
	
	avgsteps = np.mean(avgsteps,0)
	plt.plot(n,avgsteps);		
	plt.xlabel('Episodes')
	plt.ylabel('Average steps ')
	plt.title('Average Reward per episode')
	plt.grid(True)
	plt.show()

	avgreward = np.mean(avgreward,0);
	plt.plot(n,avgreward);		
	plt.xlabel('Episodes')
	plt.ylabel('Average Reward ')
	plt.title('Average Reward per episode')
	plt.grid(True)
	plt.show()
			
def printScore(run,afterEpisodes, score_tuple):
	print "%d\t\t%d\t\t%.2f\t\t%.1f" % (run, afterEpisodes, score_tuple[0],score_tuple[1])

def Startexperiment():
	sum=0;
	this_return=0;
	mean=0;
	n=1;
	steps = 0;
	avgsteps = 0;
	
	for i in range(0,n): # n trials			
		RLGlue.RL_episode(100000) # 0 indicates infinite steps
		this_return=RLGlue.RL_return();
		steps+=RLGlue.RL_num_steps();
		sum+=this_return;
	
	mean=sum/n;
	avgsteps = steps/n;
	return mean,avgsteps;


RLGlue.RL_init()
print "\nStarting SARSA on Puddle World \n-------------------------------------------------------------------------\n"
print "Trails will be conducted for 10 runs.\n"
print "Episode\t\tReturn\t\tSteps\n-------------------------------------------------------------------------"

# RLGlue.RL_agent_message("set-start-state 6 1")

# offlineDemo()

# RLGlue.RL_agent_message("save_policy results.dat");

while True:
	a = input ("waiting for the key: ")
	if (a==1):
		RLGlue.RL_env_message("set-start-state 1 0 ")
RLGlue.RL_cleanup();
print "\nProgram Complete."
