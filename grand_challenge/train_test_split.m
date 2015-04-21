function [Xtrain, Ytrain, Xtest, Ytest] = train_test_split(X, Y, split, randomize)
[n, ~] = size(X);

if randomize
    perm = randperm(n); 
else
    perm = 1:n;
end

split_i = floor(n * split);

% X
X_shuff = X(perm, :);
Xtrain = X_shuff(1:split_i-1, :);
Xtest = X_shuff(split_i:n, :);

% Y 
Y_shuff = Y(perm, :);
Ytrain = Y_shuff(1:split_i-1, :);
Ytest = Y_shuff(split_i:n, :);

end