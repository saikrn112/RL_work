% CS6700 - Reinforcement Learning Programming Assignment 1 

close all 
clc
clear all 
eps = 0.01;
temp = 0.1;
c = 2;
nA = 1000;
Nsteps = 7000;
Nruns = 500;
avgrew = [];

% E-greedy
runs = zeros(Nruns,Nsteps); 
parfor i=1:Nruns
    Qarms = zeros(1,nA);
    Npull = ones(1,nA);
    arm = normrnd(0,1,[1 nA]);
    for j=1:Nsteps
        e = rand(1);
        if(e <= eps)
            k = randi([1 nA]);
        else
           [tmp,k] = max(Qarms);
        end
        runs(i,j) = normrnd(arm(k),1); 
        Npull(k) = Npull(k)+1;
        Qarms(k) = Qarms(k) + 1/Npull(k)*(runs(i,j) - Qarms(k));
    end
end
avgrew = [avgrew; mean(runs,1)];

% Gibbs Method
runs = zeros(Nruns,Nsteps); 
parfor i=1:Nruns
    Qarms = zeros(1,nA);
    Npull = ones(1,nA);
    arm = normrnd(0,1,[1 nA]);
    for j=1:Nsteps
        num =exp(Qarms/temp);
        total = sum(num);
        probdist = num ./total; 
        [mx,k] = histc(rand(1),[0 cumsum(probdist)]);
        runs(i,j) = normrnd(arm(k),1);
        Npull(k) = Npull(k)+1;
        Qarms(k) = Qarms(k) + 1/Npull(k)*(runs(i,j) - Qarms(k));
    end
end
avgrew = [avgrew; mean(runs,1)];

%UCB
runs = zeros(Nruns,Nsteps); % 2000 runs- along the column 1000 plays - along row => Trajectory is from left to right 
parfor i=1:Nruns
    Qarms = zeros(1,nA);
    Npull = ones(1,nA);
    arm = normrnd(0,1,[1 nA]);
    for j=1:Nsteps
        if (j<=nA)
            runs(i,j) = normrnd(arm(j),1);
            Npull(j) = Npull(j)+1;
            Qarms(j) = Qarms(j) + 1/Npull(j)*(runs(i,j) - Qarms(j));
        elseif (j>nA)
            uc = zeros(size(Qarms));
            for l=1:length(Qarms)
                uc(l) = Qarms(l) + c*sqrt(log(j)/Npull(l));
            end
            [tmp, k] = max(uc);
            runs(i,j) = normrnd(arm(k),1);
            Npull(k) = Npull(k)+1;
            Qarms(k) = Qarms(k) + 1/Npull(k)*(runs(i,j) - Qarms(k));
        end
    end
end
avgrew = [avgrew; mean(runs,1)];

% Plotting
steps = linspace(1,Nsteps, Nsteps);
figure ; hold on; color = 'brkgy';
for ep = 1:3
    plot(steps,avgrew(ep,:), [color(ep),'-']);
end
legend(  { 'e-greedy', 'Gibbs','UCB' }, 'Location', 'SouthEast' ); 
title('Comparision of a UCB method with e-greedy and Gibbs for 1000 arm Bandit');
axis tight; grid on; 
xlabel( 'Steps' ); ylabel( 'Avgearge Reward for 2000 runs' );

