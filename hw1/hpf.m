function hpf_signal = hpf(signal, alpha)
    signal_len = length(signal);
    hpf_signal = zeros(signal_len - 1, 1);
    hpf_signal(1) = signal(1);
    
    for i = 2:signal_len
        hpf_signal(i) = (1-alpha) * ( ...
            hpf_signal(i-1) + ...
            signal(i) - signal(i-1) ...
        );
    end
    
end