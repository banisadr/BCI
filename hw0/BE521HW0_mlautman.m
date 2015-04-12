%%
% <latex>
% \title{BE 521 - Homework 0\\{\normalsize Spring 2015}}
% \author{Mike Lautman}
% \date{\today}
% \maketitle
% </latex>

%%
% <latex>
% \section*{1. Unit Activity}
% </latex>

%%
% <latex>
% \subsection*{1.1 IEEG portal spikes}
% 	\centering
% 	\includegraphics[width=4in]{three_peaks_IEEG.png}
% 	\\Figure 1. IEEG portal screenshot
% </latex>

%%
% <latex>
%   \subsection*{1.2 Grab data from IEEG}
% </latex>

clear; clc; clf; close all;
dataset = 'I521_A0001_D001';
me = 'mlautman';
pass_file = 'mla_ieeglogin.bin';
[T,session] = evalc('IEEGSession(dataset, me, pass_file)');

session

%%
% <latex>
% \subsection*{1.3 sample rate}
% </latex>

data=session.data;
sample_rate = data.sampleRate

%%
% <latex>
% \subsection*{1.4 recording length}
% </latex>

recording_length= data.channels(1).getNrSamples;
recording_length_s = recording_length/sample_rate

%%
% <latex>
% \subsection*{1.5a same window}
% </latex>

s_s = 8.75;
e_s = s_s + .5;
s = max(round(s_s*sample_rate), 1);
e = min(round(e_s*sample_rate), recording_length);
vals = data.getvalues(s:e,1);
figure(1);

plot((s:e)./data.sampleRate, vals, 'color', 'b');

xlim([s_s, e_s]);
ylabel('Voltage (mV)', 'FontSize',10,'FontWeight','bold');
xlabel('Time (S)', 'FontSize',10,'FontWeight','bold');
title('Three spikes from IEEG dataset I521\_A0001\_D001', 'FontSize',12,'FontWeight','bold');


%%
% <latex>
% \raggedright
% \subsection*{1.5b Spikes}
% </latex>

% NOTE: We define a spike as a locally convex region where the 
% local maxima is greater than 5*std from the mean.

figure(2)
v_ave = mean(vals);
v_std = std(vals);
vals_spikes = (vals - v_ave > 5 * v_std) .* vals;
[pks,locs] = findpeaks(vals_spikes);
hold on 


plot((s:e)./data.sampleRate, vals,'color', 'b' );
plot((s + locs)/sample_rate,pks, 'x', 'color','r');

xlim([s_s, e_s]);
ylabel('Voltage (mV)', 'FontSize',10,'FontWeight','bold');
xlabel('Time (S)', 'FontSize',10,'FontWeight','bold');
title('Three peaks from IEEG dataset I521\_A0001\_D001', 'FontSize',12,'FontWeight','bold');


%%
% <latex>
% \raggedright
% \subsection*{1.5c Total Spikes in Recording}
% </latex>

vals_all = data.getvalues(1:recording_length,1);
v_all_ave = mean(vals_all);
v_all_std = std(vals_all);
vals_all_spikes = (vals_all - v_all_ave > 5 * v_all_std) .* vals_all;
[pks_all,locs_all] = findpeaks(vals_all_spikes);
length(pks_all)

close all; clc;
