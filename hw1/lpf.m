function lpf_signal = lpf(signal, beta)
    signal_len = length(signal);
    
    lpf_signal = zeros(size(signal));
    
    lpf_signal(1) = signal(1);
    
    for i = 2:signal_len
        lpf_signal(i) = (1-beta) * lpf_signal(i-1) + ...
            beta * signal(i) ;
    end
    
end