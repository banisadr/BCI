function features = create_features(signal)
%     extract features from a 1d signal.
   
%% generate lpf filter
order = 10;
% Pass band Ripple in dB
Rp = .01;    
% Stop band Ripple in dB
Rs = 10;    
% Edge frequency in Hz
e_f = 3;  
% Normalized Edge frequency in pi * rad/sample
Wp = e_f / 240 * 2*pi;   
% Design low pass filter
[b,a] = ellip(order, Rp, Rs, Wp, 'low'); 

%%
ff = filtfilt(b, a, signal);



end