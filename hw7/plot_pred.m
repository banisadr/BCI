
figure(4)
hold on 
histogram(s_trn(:,3), 'FaceColor', 'b')
histogram(s_tst(:,3), 'FaceColor', 'r')
title('Z Positions Histogram in Testing and Training sets')
xlabel('Z Position')
ylabel('Count')
legend('Training','Testing','Location', 'best')

figure(5)
hold on 
histogram(s_trn(:,1), 'FaceColor', 'b')
histogram(s_tst(:,1), 'FaceColor', 'r')
title('X Positions Histogram in Testing and Training sets')
xlabel('X Position')
ylabel('Count')
legend('Training','Testing','Location', 'best')

figure(6)
hold on 
histogram(s_trn(:,2), 'FaceColor', 'b')
histogram(s_tst(:,2), 'FaceColor', 'r')
title('Y Positions Histogram in Testing and Training sets')
xlabel('Y Position')
ylabel('Count')
legend('Training','Testing','Location', 'best')


% plotting
figure(7)
subplot(3,1,1)
hold on 
plot((1:length(s_trn(:,1))) * .070, s_trn(:,1), 'b')
plot((1:length(u_trn(:,1))) * .070, u_trn(:,1), 'r')
title('Training X Position')
legend('True X position', 'Predicted X position', 'Location', 'best')
xlabel('Time (S)')
ylabel('X Position (Unknown units)')

subplot(3,1,2)
hold on 
plot((1:length(s_trn(:,2))) * .070, s_trn(:,2), 'b')
plot((1:length(u_trn(:,2))) * .070, u_trn(:,2), 'r')
title('Training Y Position')
legend('True Y position', 'Predicted Y position', 'Location', 'best')
xlabel('Time (S)')
ylabel('Y Position (Unknown units)')

subplot(3,1,3)
hold on 
plot((1:length(s_trn(:,3))) * .070, s_trn(:,3), 'b')
plot((1:length(u_trn(:,3))) * .070, u_trn(:,3), 'r')
title('Training Z Position')
legend('True Z position', 'Predicted Z position', 'Location', 'best')
xlabel('Time (S)')
ylabel('Z Position (Unknown units)')

figure(8)
subplot(3,1,1)
hold on 
plot((1:length(s_tst(:,1))) * .070, s_tst(:,1), 'b')
plot((1:length(u_tst(:,1))) * .070, u_tst(:,1), 'r')
title('Test X Position')
legend('True X position', 'Predicted X position', 'Location', 'best')
xlabel('Time (S)')
ylabel('X Position (Unknown units)')

subplot(3,1,2)
hold on 
plot((1:length(s_tst(:,2))) * .070, s_tst(:,2), 'b')
plot((1:length(u_tst(:,2))) * .070, u_tst(:,2), 'r')
title('Test Y Position')
legend('True Y position', 'Predicted Y position', 'Location', 'best')
xlabel('Time (S)')
ylabel('Y Position (Unknown units)')


subplot(3,1,3)
hold on 
plot((1:length(s_tst(:,3))) * .070, s_tst(:,3), 'b')
plot((1:length(u_tst(:,3))) * .070, u_tst(:,3), 'r')
title('Test Z Position')
legend('True Z position', 'Predicted Z position', 'Location', 'best')
xlabel('Time (S)')
ylabel('X Position (Unknown units)')
