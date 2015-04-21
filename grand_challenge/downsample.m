function downsampled = downsample(signal, r)
[n, d] = size(signal);
new_len = ceil(n/r);

downsampled = zeros(new_len, d);
for i = 1:d
%     downsampled(:,i) = decimate(signal(:,i), r,'fir');
    downsampled(:,i) = signal(1:r:n,i);
end
   
end