clear all; close all; clc;
addpath('../dataset/VehicleLogo-mat');
addpath('../CNN');
addpath('../util');
%% set global parameter and cnn layers
global useDebugMode;
global useBatchShuffle;
global usePreTrainModel;
global useDisplayModel;
global useDataTransform;
global useSnapshot;
global useFileLog;

useDebugMode = true;
usePreTrainModel = false;
useDisplayModel = true;
useDataTransform = false;
useSnapshot = false;
useFileLog = false;
% if exist('MaxPooling.cpp') && ~exist('MaxPooling.mexa64')
%     mex MaxPooling.cpp;
% end
% if exist('StachasticPooling.cpp') && ~exist('StachasticPooling.mexa64')
%     mex StachasticPooling.cpp;
% end
%%  optimization parameters
opts.alpha = 0.1;
opts.batchsize = 3;
opts.moment = 0.0;
opts.decay=0.001;
opts.numepochs = 1;
opts.paramfiller = 'xavier';
opts.droprate = 0.0;
opts.testinterval = 20;
opts.lossfunc = @msefunc;

%% load dataset and process it  (datameta to h*w*n*c, labelmeta to c*n)
load data48_596_501;
% --- add data process to gerneral  here --- %
train_x = double(reshape(train_x',48,48,596))/255;
test_x = double(reshape(test_x',48,48,501))/255;
train_y = double(train_y');
test_y = double(test_y');

if useDataTransform
    % --- add data transform here(sometimes you need to  )---%
    
end

%% model design and set
cnn.type = 'C';
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputchannels', 8, 'kernelsize', 5, 'stride', 1, 'activefunc', @sigm) %convolution layer
    struct('type', 's', 'poolsize', 2, 'stride', [2, 2], 'poolmethod', 'a') %sub sampling layer
    struct('type', 'c', 'outputchannels', 12, 'kernelsize',3, 'stride', 1, 'activefunc', @sigm) %convolution layer
    struct('type', 's', 'poolsize', 2, 'stride', 2, 'poolmethod', 'a') %sub sampling layer
    struct('type', 'c', 'outputchannels', 16, 'kernelsize',3, 'stride', 1, 'activefunc', @sigm) %convolution layer
    struct('type', 's', 'poolsize', 2, 'stride', 2, 'poolmethod', 'a') %sub sampling layer
    struct('type', 'fc', 'outputchannels', 18, 'activefunc', @sigm)
    };
cnn = cnnsetup(cnn, opts, train_x, train_y);
if useDebugMode
    load cnn-debug
    cnn = cnnsetupwithmodel(cnn,cnninit);
end

%% train and test
cnnshowoptparam(opts);
cnnshowmodel(cnn,opts);
if useDebugMode
    useBatchShuffle = false;
    cnn = cnntrain(cnn, opts, train_x(:,:,1:10), train_y(:,1:10), test_x, test_y);
else % normal mode
    useBatchShuffle = true;
    cnn = cnntrain(cnn, opts, train_x, train_y, test_x, test_y);
end
trainrslt = cnntest(cnn, opts, train_x, train_y);
testrslt = cnntest(cnn, opts, test_x, test_y);
cnnshowresult(trainrslt, opts, train_x,train_y);
cnnshowresult(testrslt, opts, test_x,test_y);

%% save model and display
save cnn cnn;
plot(cnn.rl);
title('logo');
saveas(gcf,'rl.png');

%close all;


