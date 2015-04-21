function features = gen_features(signal, window, r)
[n, d] = size(signal);
new_size = ceil(n/r);
Fs = 1000;
num_feats = 6;

features = zeros(new_size, d * num_feats);
for j = 1:new_size
    s = (j-1) * r + 1;
    e = j * r + window;
    
    for i=1:d
%         snip = signal(s:min(n,e),i);
        features(j, (i-1) * num_feats + 1) = mean(signal(s:min(n,e),i));

        NFFT = length(signal(s:min(n,e),i));
        Y = fft(signal(s:min(n,e),i), NFFT);
        F = abs(((0:1/NFFT:1-1/NFFT)*Fs).'-Fs/2);
        features(j, (i-1) * num_feats + 2) = ...
            sum(Y(find(F>=5 & F<15)));
        features(j, (i-1) * num_feats + 3) = ...
            sum(Y(find(F>=20 & F<25)));
        features(j, (i-1) * num_feats + 4) = ...
            sum(Y(find(F>=75 & F<115)));
        features(j, (i-1) * num_feats + 5) = ...
            sum(Y(find(F>=125 & F<160)));
        features(j, (i-1) * num_feats + 6) = ...
            sum(Y(find(F>=160 & F<175)));
    end
end




% end