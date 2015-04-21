function downsampled = downsample(signal, downsamp, offset)
[n, ~] = size(signal);
% new_len = ceil(n/r);

% downsampled = zeros(new_len, d);
% for i = 1:d
% %     downsampled(:,i) = decimate(signal(:,i), r,'fir');
%     downsampled(:,i) = signal(r:r:n,i);
% end
downsampled = signal(downsamp:downsamp:n,:);
[n, ~] = size(downsampled);
downsampled = downsampled(offset:n,:);

end