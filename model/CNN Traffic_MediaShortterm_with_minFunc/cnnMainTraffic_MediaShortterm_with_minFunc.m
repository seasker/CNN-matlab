clear all;close all;clc;
addpath('../../dataset/Traffic-original-data-mat');
addpath(genpath('../../minFunc_2012'));
addpath('../../util');
addpath('../../CNN_with_minFunc_new');
%% set global parameter [needed]
global usePreTrainModel;
global useDisplayModel;
global useDataTransform;
global useBatchShuffle;
global useSnapshot;
global useFileLog;

usePreTrainModel = false;
useDisplayModel = true;
useDataTransform = true;
useBatchShuffle = true;
useSnapshot= true;
useFileLog = false;
%%  optimization parameters
opts.alpha = 0.3;
opts.batchsize =10000;
opts.moment = 0.9;
opts.decay=0.001;
opts.numepochs = 40;
opts.paramfiller = 'xavier';
opts.droprate = 0.0;
opts.testinterval = 1;
opts.snapshotinterval = 20;
opts.lossfunc = @msefunc;
minfuncopts.Method = 'scg';
minfuncopts.Display ='Iter';
minfuncopts.MaxIter = 5;
%% load dataset and process it  (datameta to h*w*n*c, labelmeta to c*n)
load trafficcontinousFSOdata
trainnum = 93600;
transmeta.maxval = 1;
transmeta.minval = 0;
[train_x,train_y,test_x,test_y, datameta] = datapreprocess(trafficcontinousFSOdata,trainnum);
[train_x,train_y] = datatransform(train_x, train_y, datameta, transmeta);
[test_x,test_y] = datatransform(test_x, test_y, datameta, transmeta);
%% set cnn_traffic layers and parameters
if   usePreTrainModel
    assert(logical(exist('cnn_traffic.mat','file')),'ERROR: not exist cnn model');
    load cnn_traffic;
    cnn_traffic.stage = cnn_traffic.stage + 1;
else
    cnn_traffic.type = 'R';
    cnn_traffic.stage = 1;
    cnn_traffic.layers = {
        struct('type', 'i') %input layer
        struct('type', 'c', 'outputchannels', 8, 'kernelsize', 5 ,'stride', 1, 'activefunc', @sigm) %convolution layer
        struct('type', 's', 'poolsize', 2, 'stride', 2, 'poolmethod', 'm') %sub sampling layer
        struct('type', 'c', 'outputchannels', 16, 'kernelsize',3, 'stride', 1, 'activefunc', @sigm) %convolution layer
        struct('type', 's', 'poolsize', 2, 'stride', 2, 'poolmethod', 'm') %sub sampling layer
        struct('type', 'c', 'outputchannels', 32, 'kernelsize',3, 'stride', 1, 'activefunc', @sigm) %convolution layer
        struct('type', 's', 'poolsize', 2, 'stride', 2, 'poolmethod', 'm') %sub sampling layer
        %struct('type', 'fc','outputchannels', 'activefunc', @sigm)
        struct('type', 'fc', 'activefunc', @sigm)
        };
    cnn_traffic = cnnsetup(cnn_traffic, opts, train_x, train_y );
end
%% train and test model
cnnshowoptparam(opts);
cnnshowmodel(cnn_traffic);
cnn_traffic = cnntrainwithminfunc(cnn_traffic, opts, train_x, train_y, test_x, test_y,minfuncopts);
trainrslt = cnntest(cnn_traffic,opts, train_x, train_y);
testrslt = cnntest(cnn_traffic, opts,test_x, test_y);
cnnshowoptparam(opts);
cnnshowmodel(cnn_traffic);
cnnshowresult(trainrslt, opts, train_x, train_y, datameta, transmeta);
cnnshowresult(testrslt, opts, test_x, test_y,  datameta, transmeta);

useFileLog = true;
if isfield(opts,'logfilename')
    fid = fopen(opts.logfilename,'a+');
else
    fid = fopen('result.txt', 'a+');
end
fprintf(fid,'%s-net-stage-%d\n',date, cnn_traffic.stage);
fclose(fid);
cnnshowoptparam(opts);
if cnn_traffic.stage == 1
    cnnshowmodel(cnn_traffic, opts);
end
cnnshowresult(trainrslt, opts, train_x, train_y, datameta, transmeta);
cnnshowresult(testrslt, opts, test_x, test_y,  datameta, transmeta);
%% save model and display
save ('cnn_traffic','cnn_traffic','-v7.3');
plot(cnn_traffic.rl);
title('traffic');
legend('last batch optimized loss','smooth batch optimized loss','last batch loss','smooth batch loss');
saveas(gcf,'rl.png');
%close all;
