function feat_new = add_history(features, hist)
% N backwards looking time stamps for each feature for feature vect
[n, d] = size(features);

d_new = d * hist + 1; % features * time steps plus a ones vect for bias
n_new = size(features,1) - hist + 1;

feat_new = zeros(n_new, d_new);

for i=1 : n_new
    e = i + hist - 1;
    row = features(i:e, :);
    feat_new(i, :) = [1, row(:)'];
end

end