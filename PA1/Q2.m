function [] = Q2()
%% Question - 2 
% To make Assignment more modular 3 functions were created 
% egreedy
% gibbs <------ 
% ucb

% Removing all unnecessary windows and variables 
close all 
clc
clear all 

truemu = 0; % True mean distirbution mean 
truesigma = 1; % True mean distribution variance// sigma

Narms = 10; % Number of arms per bandit 
Nsteps = 1000; % Number of steps/ run 
Nruns = 2000; % Number of runs
avgrew = []; 

temp = [0.01 0.1 1 1.2 1000 ]; % Different epsilons for which experiments were conducted

for tp = 1:length(temp) % Looping for different epsilons 
    %% Variable initialisation for different epsilon calculations 
    Qarms = zeros(Nruns,Narms); % 2D matrix for different Qvalues
    Npull = ones(Nruns,Narms); % 2D matrix for different Number of pulls of a particular arm
    runs = zeros(Nruns,Nsteps); % 2D matrix for different Runs   
    for i=1:Nruns
        arm = normrnd(truemu,truesigma,[1 Narms]); % arm variable defines the Truemean distribution 
        for j=1:Nsteps
            %% Action selection and update
            %k = egreedy(Qarms(i,:),ep); % Selecting the arm based on egreedy 
            k = gibbs(Qarms(i,:),temp(tp)); % Selecting the arm based on Gibbs 
            %k = ucb(Qarms(i,:),j,i); % Selecting the arm based on UCB
            
            runs(i,j) = normrnd(arm(k),1);  % Reward from pulling arm k
            Npull(i,k) = Npull(i,k)+1; % Counting the pulls 
            Qarms(i,k) = Qarms(i,k) + 1/Npull(i,k)*(runs(i,j) - Qarms(i,k)); % Incremental Update of Q value of an arm            
        end
    end 
    %% Storing the results obtained for different epsilons
    avgrew = [avgrew; mean(runs,1)];
end

%% Plotting Average reward
steps = linspace(1,Nsteps,Nsteps);
figure ; hold on; grid on ; color = 'brkcyrg';
for tp = 1:length(temp)
plot(steps,avgrew(tp,:), [color(tp),'-']);
end
legend(  {'0.01','0.1','1','1.2','1000' }, 'Location', 'SouthEast' ); 
title('Average Performance of a Gibbs Softmax method on 10 arm bandits');
axis tight; grid on; 
xlabel( 'Steps' ); ylabel( 'Avgearge Reward for 2000 runs' );


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