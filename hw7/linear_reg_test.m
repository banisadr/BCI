
%% Linear regression 
load('HW7Data');
n_neur = size(test_count, 2);
N = 5; % backwards looking time stamps for each neuron for feature vect
d = n_neur * N + 1; % 40 neurons by 20 time steps plus a ones vect for bias
Mtrn = size(count,1) - N + 1;
R_trn = zeros(Mtrn, d);

Mtst = size(test_count,1) - N + 1; % last timestamp for positions
R_tst = zeros(Mtst, d);

scale = 1; % Scaling discretization bins
rand_test_set = false; 
% We create the R trn matrix

for i=1:Mtrn
    e = i + N - 1;
    row = count(i:e, :);
    R_trn(i, :) = [1, row(:)'];
end

% We create the R tst matrix
for i=1:Mtst
    e = i + N - 1;
    row = test_count(i:e, :);
    R_tst(i, :) = [1, row(:)'];
end

% Randomize test set (or not)
if rand_test_set
    perm = randperm(size(R,1)); 
else
    perm = 1:size(R_trn,1);
end

% training set
R_trn = R_trn(perm, :);

% Training data NOT discretized 
s_trn = angles(perm, :);

% Linear regression explicit solution
f_trn = pinv(R_trn' * R_trn) * (R_trn' * s_trn);

% predictions
u_tst = R_tst * f_trn;
u_trn = R_trn * f_trn;

% Solving for the correlation scores
rho2_x = corr(s_trn(:,1),u_trn(:,1))
rho2_y = corr(s_trn(:,2),u_trn(:,2))
rho2_z = corr(s_trn(:,3),u_trn(:,3))
    
c = corr(s_trn, u_trn)

%% Find best lambda x using CV
% [Bx2_n,FitInfo1] = lasso(R_trn(:,2:size(R_trn,2)),s_trn(:,1),'NumLambda',5, 'CV', 5);
% lassoPlot(Bx2_n,FitInfo1,'PlotType','CV');

%% Build model for X
[Bx,FitInfo_x] = lasso(R_trn(:,2:size(R_trn,2)), s_trn(:,1),'Lambda', 0.015);
Bo = FitInfo_x.Intercept;
Bx = [Bo;Bx];
predX_trn = R_trn * Bx;
predX = R_tst * Bx;
% rho_x = corr(s_tst(:,1), predX)

%% Find best lambda y through CV
% [By2_n,FitInfo1] = lasso(R_trn(:,2:size(R_trn,2)),s_trn(:,2),'NumLambda', 10, 'CV', 10);
% lassoPlot(By2_n,FitInfo1,'PlotType','CV');

%% Build model for Y
[By,FitInfo_y] = lasso(R_trn(:, 2:size(R_trn,2)), s_trn(:,2),'Lambda', 0.015);
Bo = FitInfo_y.Intercept;
By = [Bo; By];
predY_trn = R_trn * By;
predY = R_tst * By;
% rho_y = corr(s_tst(:,2), predY)

%% Find best lambda z through CV
% [By2_n,FitInfo1] = lasso(R_trn(:,2:size(R_trn,2)),s_trn(:,2),'NumLambda',10, 'CV', 10);
% lassoPlot(By2_n,FitInfo1,'PlotType','CV');

%% Build model for Z
[Bz,FitInfo_z] = lasso(R_trn(:,2:size(R_trn,2)), s_trn(:,3),'Lambda',0.015);
Bo = FitInfo_z.Intercept;
Bz = [Bo ;Bz];
predZ_trn = R_trn * Bz;
predZ = R_tst * Bz;
% rho_z = corr(s_tst(:,3), predZ)


%% Animate results
figure
subplot(1,3,1);
hold on 
axis([...
	min(s_trn(:,1)) max(s_trn(:,1)) ...
	min(s_trn(:,2)) max(s_trn(:,2)) ...
	min(s_trn(:,3)) max(s_trn(:,3)) ...
])
subplot(1,3,2);
hold on 
axis([...
	min(predX_trn) max(predX_trn) ...
	min(predY_trn) max(predY_trn) ...
	min(predZ_trn) max(predZ_trn) ...
])
subplot(1,3,3);
hold on 
axis([...
	min(predX) max(predX) ...
	min(predY) max(predY) ...
	min(predZ) max(predZ) ...
])
for i=1:min(length(predX), length(predX_trn))
    subplot(1,3,1);
	scatter3(s_trn(i,1), s_trn(i,2), s_trn(i,3))

    subplot(1,3,2);
    scatter3(predX_trn(i), predY_trn(i), predZ_trn(i))

    subplot(1,3,3);
	scatter3(predX(i), predY(i), predZ(i))
	pause(.01)
end