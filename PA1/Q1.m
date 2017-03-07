function [avgrew,optimal_arm] = Q1()
%% CS6700 - Reinforcement Learning Programming Assignment 1 
% Question - 1 
% To make Assignment more modular 3 functions were created 
% egreedy <-----------
% gibbs 
% ucb

% Removing all unnecessary windows and variables 
close all 
clc
clear all 


eps = [0,0.01,0.1,0.3,0.5,1]; % Different epsilons for which experiments were conducted
truemu = 0; % True mean distirbution mean 
truesigma = 1; % True mean distribution variance// sigma

Narms = 10; % Number of arms per bandit 
Nsteps = 1000; % Number of steps/ run 
Nruns = 2000; % Number of runs

avgrew = []; optimal_arm = [];

for ep = 1:length(eps) % Looping for different epsilons 
    %% Variable initialisation for different epsilon calculations 
    Qarms = zeros(Nruns,Narms); % 2D matrix for different Qvalues
    Npull = ones(Nruns,Narms); % 2D matrix for different Number of pulls of a particular arm
    runs = zeros(Nruns,Nsteps); % 2D matrix for different Runs
    optiactions = zeros(Nruns, Nsteps); %2D matrix for storing when was the optimal action pulled
    
    
    for i=1:Nruns
        arm = normrnd(truemu,truesigma,[1 Narms]); % arm variable defines the Truemean distribution 
        [tmp, bestarm] = max(arm); % BestArm in the current distribution; tmp is dummy variable
        for j=1:Nsteps
            %% Action selection and Update
            k = egreedy(Qarms(i,:),eps(ep)); % Selecting the arm based on egreedy 
            %k = gibbs(Qarms(i,:),temp(tp)); % Selecting the arm based on Gibbs 
            %k = ucb(Qarms(i,:),j,i); % Selecting the arm based on UCB
            
            runs(i,j) = normrnd(arm(k),1);  % Reward from pulling arm k
            Npull(i,k) = Npull(i,k)+1; % Counting the pulls 
            Qarms(i,k) = Qarms(i,k) + 1/Npull(i,k)*(runs(i,j) - Qarms(i,k)); % Incremental Update of Q value of an arm
            %% Optimal Action check
            % Incrementing the value in Optiactions if the best arm is
            % pulled at this instant
            if(k==bestarm)
                optiactions(i,j) = 1; 
            end
        end
    end
    
    %% Storing the results obtained for different epsilons
    optimal_arm = [optimal_arm;mean(optiactions,1)]; % Number of Times Optimal arm is chosen; Concatinating with previous value 
    avgrew = [avgrew; mean(runs,1)]; % Average reward for 2000 runs; Concatinating with previous value
end

%% Plotting Average Reward
steps = linspace(1,Nsteps, Nsteps);
figure ; hold on; color = 'brkgym';
for ep = 1:length(eps)
    plot(steps,avgrew(ep,:), [color(ep),'-']);
end
legend(  {'0', '0.01', '0.1','0.3', '0.5','1' }, 'Location', 'SouthEast' ); 
title('Average Performance of a e-greedy method on 10 arm bandits');
axis tight; grid on; 
xlabel( 'Steps' ); ylabel( 'Avgearge Reward for 2000runs' );

%% Plotting Optimal Actions for 2000 runs
figure ; hold on; grid on;
for ep = 1:length(eps)
    plot(steps,optimal_arm(ep,:)*100, [color(ep),'-']);
end
legend(  {'0', '0.01', '0.1','0.3','0.5', '1' }, 'Location', 'SouthEast' ); 
title('Number of times Optimal actions taken by a e-greedy method on 10 arm bandits ');
axis tight; grid on; 
xlabel( 'Steps' ); ylabel( 'Optimal actions %' );

function [k] = egreedy(Q,eps) 
%% Egreedy method
e = rand(1);
if(e <= eps) % If random number is less than epsilon select action randomly
    k = randi([1 10]);
else % If random number is greater than epsilon select action greedily
   [tmp,k] = max(Q); %tmp is a dummy variable used to extract the index of maximum value in Q 
end

function [k] = gibbs(Q,temp)
%% Gibbs method
num =exp(Q/temp); % calculate the exponential of Q values
total = sum(num); % Sum them up
probdist = num ./total; % Gibbs probability Distribution
% Construct a set of bins with their probabilities acting as weights 
% These weights are nothing but gibbs distribution of that particular
% action; High probability action has more weight therefore high
% probability of random number falling into that bin
[tmp,k] = histc(rand(1),[0 cumsum(probdist)]);  %tmp is a dummy variable used to extract the index of maximum value in Q 


function [k] = ucb(Q,j,l)
%% UCB method
global Npull   
c = 2; % Confidence factor
uc = zeros(size(Q)); % Initializing a new array to store the Upper confidence bounds
for i=1:length(Q)
    uc(i) = Q(i) + c*sqrt(log(j)/Npull(l,i));
end
[tmp, k] = max(uc); %tmp is a dummy variable used to extract the index of maximum value in Q 