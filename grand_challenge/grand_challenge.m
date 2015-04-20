clear; clc;

train_dataset ='I521_A0012_D001';
train_labels = 'I521_A0012_D002';
me = 'mlautman';
pass_file = 'mla_ieeglogin.bin';
[T, session_label] = evalc('IEEGSession(train_labels, me, pass_file)');
[T, session_train] = evalc('IEEGSession(train_dataset, me, pass_file)');

data_label = session_label.data;
data_train = session_train.data;

sr_label = data_label.sampleRate;
sr_train = data_train.sampleRate;

nr_label = session_label.data.channels(1).getNrSamples;

labels = cell(1,5);
for i = 1:5
    labels{i} = data_label.getvalues(1:nr_label,i);
end

[T, session_train] = evalc('IEEGSession(train_dataset, me, pass_file)');
data_train = session_train.data;
sr_train = data_train.sampleRate;
nr_train = session_train.data.channels(1).getNrSamples;
train = cell(1,5);
for i = 1:5
    train{i} = data_train.getvalues(1:nr_train,i);
end
