function corr_scores = compute_corr_loss_downsample(signal, r) 

% test the correlation to the initial signal when downsampled with decimate
downsampled = downsample(signal,r);
recovered = zeros(size(signal));
for i=1:size(downsampled,1)
    recovered((i-1)* r + 1:i*r,:) = bsxfun(@times, downsampled(i,:) ,ones(r,1));
end

corr_scores = zeros(1,size(downsampled,2)); 
for i = 1:size(downsampled,2)
    corr_scores(i) = corr(recovered(:,i), signal(:,i));
end

end