function data = load_data(base)

me = 'mlautman';
pass_file = 'mla_ieeglogin.bin';

data = struct();
data.base = base;
data.train = struct();
data.test = struct();

data.train.ecog_fid = strcat(base, '_D001');
data.train.label_fid = strcat(base, '_D002');
data.test.ecog_fid = strcat(base, '_D003');

[~, data.train.ecog_conn ] = evalc(...
    'IEEGSession(data.train.ecog_fid, me, pass_file)');
[~, data.train.label_conn ] = evalc(...
    'IEEGSession(data.train.label_fid, me, pass_file)');
[~, data.test.ecog_conn ] = evalc(...
    'IEEGSession(data.test.ecog_fid, me, pass_file)');

data.train.sr = data.train.ecog_conn.data.sampleRate;
sr_label = data.train.label_conn.data.sampleRate;
if data.train.sr ~= sr_label 
    warning('train sample rate mismatch')
end

data.train.nr_samples = data.train.ecog_conn.data.channels(1).getNrSamples + 1;

data_channels = size(data.train.ecog_conn.data.channels,2);
label_channels = size(data.train.label_conn.data.channels,2);
data.train.ecog = data.train.ecog_conn.data.getvalues(1:data.train.nr_samples,1:data_channels);
data.train.label = data.train.label_conn.data.getvalues(1:data.train.nr_samples,1:label_channels);

data.test.sr = data.test.ecog_conn.data.sampleRate;
if data.test.sr ~= data.train.sr 
    warning('test-train sample rate mismatch')
end

data_channels = size(data.test.ecog_conn.data.channels,2);
data.test.nr_samples = data.test.ecog_conn.data.channels(1).getNrSamples + 1;
data.test.ecog = data.test.ecog_conn.data.getvalues(1:data.test.nr_samples,1:data_channels); 

end
