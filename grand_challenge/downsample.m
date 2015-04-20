function downsampled = downsample(signal, r)

new_len = ceil(size(signal,1)/r);
d = size(signal,2);
downsampled = zeros(new_len, d);
for i = 1:d
    downsampled(:,i) = decimate(signal(:,i), r);
end
    
end