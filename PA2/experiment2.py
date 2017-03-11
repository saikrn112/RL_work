

import sys
import math
import rlglue.RLGlue as RLGlue
import matplotlib.pyplot as plt
import numpy as np

def offlineDemo():
	statistics=[];
	episodeResult=Startexperiment();
	printScore(0,episodeResult);
	statistics.append(episodeResult);
	episodes = 1000	
	n = np.arange(1,episodes+2,1);

	for j in range(0,1): # 50 runs 
		for i in range(0,episodes): # Let's try 100 episodes for learning 
			RLGlue.RL_env_message("set-random-start-state");
			episodeResult=Startexperiment();
			printScore((i+1),episodeResult);
			statistics.append(episodeResult);

	plt.plot(n,[x[0] for  x in statistics]);		
	plt.show()
			
def printScore(afterEpisodes, score_tuple):
	print "%d\t\t%.2f\t\t%.1f" % (afterEpisodes, score_tuple[0],score_tuple[1])

def Startexperiment():
	sum=0;
	this_return=0;
	mean=0;
	n=1;
	steps = 0;
	avgsteps = 0;
	
	for i in range(0,n): # n trials			
		RLGlue.RL_episode(0) # 0 indicates infinite steps
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

offlineDemo()

RLGlue.RL_agent_message("save_policy results.dat");

# input ("waiting for the key")

RLGlue.RL_cleanup();
print "\nProgram Complete."
