% CS6700 - Reinforcement Learning Programming Assignment 1 

%% Question - 3 
% To make Assignment more modular 3 functions were created 
% egreedy 
% gibbs 
% UCB <-----------

% Removing all unnecessary windows and variables 
close all 
clc
clear all 

truemu = 0; % True mean distirbution mean 
truesigma = 1; % True mean distribution variance// sigma

Narms = 1000; % Number of arms per bandit 
Nsteps = 7000; % Number of steps/ run 
Nruns = 50; % Number of runs

eps = 0.01; % Exploration in e-greedy
temp = 0.1; % Temperature in Gibbs
c = 2; % Confidence factor

avgrew = [];

%% E-greedy
runs = zeros(Nruns,Nsteps); 
parfor i=1:Nruns
    %% Variables initialisation for different epsilon calculations
    Qarms = zeros(1,Narms); % 1D matrix for different Qvalues
    Npull = ones(1,Narms); % 1D matrix for different Number of pulls of a particular arm
    arm = normrnd(truemu,truesigma,[1 Narms]); % arm variable defines the Truemean distribution 
    for j=1:Nsteps
        %% Action selection and Update
        e = rand(1);
        if(e <= eps) % If random number is less than epsilon select action randomly
            k = randi([1 10]);
        else % If random number is greater than epsilon select action greedily
           [tmp,k] = max(Qarms); %tmp is a dummy variable used to extract the index of maximum value in Q 
        end
        runs(i,j) = normrnd(arm(k),1);  % Reward from pulling arm k
        Npull(k) = Npull(k)+1; % Counting the pulls 
        Qarms(k) = Qarms(k) + 1/Npull(k)*(runs(i,j) - Qarms(k)); % Incremental Update of Q value of an arm
    end
end
avgrew = [avgrew; mean(runs,1)];

%% Gibbs Method
runs = zeros(Nruns,Nsteps); 
parfor i=1:Nruns
    %% Variables initialisation for different epsilon calculations
    Qarms = zeros(1,Narms); % 1D matrix for different Qvalues
    Npull = ones(1,Narms); % 1D matrix for different Number of pulls of a particular arm
    arm = normrnd(truemu,truesigma,[1 Narms]); % arm variable defines the Truemean distribution 
    for j=1:Nsteps
        %% Action selection and update
        num =exp(Qarms/temp); % calculate the exponential of Q values
        total = sum(num); % Sum them up
        probdist = num ./total; % Gibbs probability Distribution
        % Construct a set of bins with their probabilities acting as weights 
        % These weights are nothing but gibbs distribution of that particular
        % action; High probability action has more weight therefore high
        % probability of random number falling into that bin
        [tmp,k] = histc(rand(1),[0 cumsum(probdist)]);  %tmp is a dummy variable used to extract the index of maximum value in Q 


        runs(i,j) = normrnd(arm(k),1);  % Reward from pulling arm k
        Npull(k) = Npull(k)+1; % Counting the pulls 
        Qarms(k) = Qarms(k) + 1/Npull(k)*(runs(i,j) - Qarms(k)); % Incremental Update of Q value of an arm
    end
end
avgrew = [avgrew; mean(runs,1)];

%% UCB
runs = zeros(Nruns,Nsteps); 
parfor i=1:Nruns
    %% Variables initialisation for different epsilon calculations
    Qarms = zeros(1,Narms); % 1D matrix for different Qvalues
    Npull = ones(1,Narms); % 1D matrix for different Number of pulls of a particular arm
    arm = normrnd(truemu,truesigma,[1 Narms]); % arm variable defines the Truemean distribution 
    for j=1:Nsteps
        if (j<=Narms)
            runs(i,j) = normrnd(arm(j),1);
            Npull(j) = Npull(j)+1;
            Qarms(j) = Qarms(j) + 1/Npull(j)*(runs(i,j) - Qarms(j));
        elseif (j>Narms)
            %% Action selection and Update
            uc = zeros(size(Qarms));  % Initializing a new array to store the Upper confidence bounds
            for l=1:length(Qarms)
                uc(l) = Qarms(l) + c*sqrt(log(j)/Npull(l));
            end
            [tmp, k] = max(uc); %tmp is a dummy variable used to extract the index of maximum value in Q 
            runs(i,j) = normrnd(arm(k),1);  % Reward from pulling arm k
            Npull(k) = Npull(k)+1; % Counting the pulls 
            Qarms(k) = Qarms(k) + 1/Npull(k)*(runs(i,j) - Qarms(k)); % Incremental Update of Q value of an arm
        end
    end
end
avgrew = [avgrew; mean(runs,1)];

%% Plotting
steps = linspace(1,Nsteps, Nsteps);
figure ; hold on; color = 'brkgy';
for ep = 1:3
    plot(steps,avgrew(ep,:), [color(ep),'-']);
end
legend(  { 'e-greedy', 'Gibbs','UCB' }, 'Location', 'SouthEast' ); 
title('Comparision of a UCB method with e-greedy and Gibbs for 1000 arm Bandit');
axis tight; grid on; 
xlabel( 'Steps' ); ylabel( 'Avgearge Reward for 2000 runs' );