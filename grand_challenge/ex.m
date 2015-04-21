base = 'I521_A0012';
window = 100;
overlap = 40;
downsamp = 40;
hist = 4;

data = load_data(base);

data.train.features = gen_features(...
    data.train.ecog, window, overlap);

data.train.features_hist = add_history(data.train.features, hist);

data.train.label_hist = downsample(data.train.label, downsamp, hist);

[Xtrain, Ytrain, Xtest, Ytest] = train_test_split(data.train.features_hist, data.train.label_hist, .5, true);


svm1 = svmtrain(Ytrain(:,1),Xtrain, '-s 0 -t 2 -g .01 -c 100.5 -w0 .01');
[pred, acc, ~ ] = svmpredict(Ytest(:,1), Xtest, svm1);
