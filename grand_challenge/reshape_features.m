%% epsilon SVC 
% load('HW7Data');
n_neur = size(features,2);
N = 5; % backwards looking time stamps for each neuron for feature vect
d = n_neur * N + 1; % 40 neurons by 20 time steps plus a ones vect for bias
Mtrn = size(features,1) - N + 1;
R_trn = zeros(Mtrn, d);
% 
% Mtst = size(test_count,1) - N + 1; % last timestamp for positions
% R_tst = zeros(Mtst, d);
% 
% scale = 1; % Scaling discretization bins
rand_test_set = false; 
% We create the R trn matrix

for i=1:Mtrn
    e = i + N - 1;
    row = features(i:e, :);
    R_trn(i, :) = [1, row(:)'];
end

% % We create the R tst matrix
% for i=1:Mtst
%     e = i + N - 1;
%     row = test_count(i:e, :);
%     R_tst(i, :) = [1, row(:)'];
% end

% Randomize test set (or not)
if rand_test_set
    perm = randperm(size(R,1)); 
else
    perm = 1:size(R_trn,1);
end

% training set
R_trn = R_trn(perm, :);

% Training data NOT discretized 
s_trn = downsampled(perm, :);
% garbage_pred = ones(size(R_tst,1),1);

% %% Useful functions for tuning parameters
% svmX = svmtrain(round(s_trn(:,1)),R_trn, '-s 4 -t 2 -c 100.5 -v 10 -q');
% svmY = svmtrain(round(s_trn(:,2)),R_trn, '-s 4 -t 2 -c 100.5 -v 10 -q');
% svmZ = svmtrain(round(s_trn(:,3)),R_trn, '-s 4 -t 2 -c 100.5 -v 10 -q');
% 
% %% Train SVR X 
% svmX = svmtrain(round(s_trn(:,1)),R_trn, '-s 3 -t 2 -c 100.5 -q');
% % make predictions on resulting set
% predX_trn = svmpredict(s_trn(:,1), R_trn, svmX, '-q');
% predX = svmpredict(garbage_pred, R_tst, svmX, '-q');
% % svr_corr_X = corr(predX, s_tst(:,1))
% % Train SVR Y 
% svmY = svmtrain(round(s_trn(:,2)),R_trn, '-s 3 -t 2 -c 100.5 -q');
% % make predictions on resulting set
% predY_trn = svmpredict(s_trn(:,2), R_trn, svmY, '-q');
% predY = svmpredict(garbage_pred, R_tst, svmY, '-q');
% % svr_corr_Y = corr(predY, s_tst(:,2))
% % Train SVR Z
% svmZ = svmtrain(round(s_trn(:,3)),R_trn, '-s 3 -t 2 -c 100.5 -q');
% % make predictions on resulting set
% predZ_trn = svmpredict(s_trn(:,3), R_trn, svmZ, '-q');
% predZ = svmpredict(garbage_pred, R_tst, svmZ, '-q');
% % svr_corr_Z = corr(predZ, s_tst(:,3))
% 
% 
% %% Animate results
% figure
% subplot(1,3,1);
% hold on 
% axis([...
% 	min(s_trn(:,1)) max(s_trn(:,1)) ...
% 	min(s_trn(:,2)) max(s_trn(:,2)) ...
% 	min(s_trn(:,3)) max(s_trn(:,3)) ...
% ])
% subplot(1,3,2);
% hold on 
% axis([...
% 	min(predX_trn) max(predX_trn) ...
% 	min(predY_trn) max(predY_trn) ...
% 	min(predZ_trn) max(predZ_trn) ...
% ])
% subplot(1,3,3);
% hold on 
% axis([...
% 	min(predX) max(predX) ...
% 	min(predY) max(predY) ...
% 	min(predZ) max(predZ) ...
% ])
% for i=1:min(length(predX), length(predX_trn))
%     subplot(1,3,1);
% 	scatter3(s_trn(i,1), s_trn(i,2), s_trn(i,3))
% 
%     subplot(1,3,2);
%     scatter3(predX_trn(i), predY_trn(i), predZ_trn(i))
% 
%     subplot(1,3,3);
% 	scatter3(predX(i), predY(i), predZ(i))
% 	pause(.01)
% end
