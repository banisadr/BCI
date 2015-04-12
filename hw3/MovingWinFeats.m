function [feature] = MovingWinFeats(x, fs, winLen, winDisp, featFn)
    windows = @(xLen, fs, winLen, winDisp) ...
        ((xLen/fs)-winLen)/winDisp + 1; 
    
    windows = windows(length(x), fs, winLen, winDisp); 
    
    feature = zeros(windows, 1); 
    
    stp = winDisp*fs; 
    len = winLen*fs; 
    
    for i=1:windows
        s= (i-1) * stp + 1;
        e = (i-1) * stp + len; 
        feature(i) = featFn(x(s:e)); 
    end
end
