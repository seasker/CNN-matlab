function net = cnntrain(net, opts, x, y, test_x , test_y)
global useBatchShuffle;
global useSnapshot;
global useFileLog;
if nargin > 4
    USETESTDATA = true;
end
net.phase = 'train';
m = size(x, 3); % m
numbatches = double(ceil(m / opts.batchsize));
batchstartidx = 1;
net.rl = [];
opts.logfilename = [net.name,'_train_process.txt'];
for epoch = 1 : opts.numepochs
    if rem(epoch, opts.testinterval)==1
        
        trainrslt = cnntest(net, opts, x, y);
        cnnshowresult(trainrslt, opts, x, y);
        useFileLog = true;
        cnnshowresult_new(trainrslt, opts, x, y, opts.datameta, opts.transmeta);
        useFileLog =false;
        if USETESTDATA
            testrslt = cnntest(net, opts, test_x, test_y);
            cnnshowresult(testrslt, opts, test_x, test_y);
            useFileLog = true;
            cnnshowresult_new(testrslt, opts, test_x, test_y,opts.datameta,opts.transmeta);
            useFileLog = false;
        end
    end
    fprintf('epoch %d/%d  ', epoch, opts.numepochs);
    tic;
    if useBatchShuffle
        order = randperm(m);
    else
        order = 1 : m;
    end
    for batch = 1 : numbatches
        batchendidx = batchstartidx + opts.batchsize - 1;
        if batchendidx > m
            batch_x(:, :, 1:m-batchstartidx+1, :) = x(:, :, order(batchstartidx:m), :);
            batch_x(:, :, m-batchstartidx+2:opts.batchsize, :) = x(:, :, order(1:batchendidx-m), :);
            batch_y(:, 1:m-batchstartidx+1) = y(:, order(batchstartidx:m));
            batch_y(:, m-batchstartidx+2:opts.batchsize) = y(:, order(1:batchendidx-m)); 
        else
            batch_x = x(:, :, order(batchstartidx:batchendidx), :);
            batch_y = y(:,    order(batchstartidx:batchendidx));
        end
        batchstartidx = mod(batchendidx, m) + 1;
        net = cnnff(net, opts, batch_x, batch_y);  % Feedforward
        net = cnnbp(net);                    % Backpropagation
        net = cnnapplygrads(net, opts);
        if isempty(net.rl)
            net.rl(1,1) = net.optmloss;
            net.rl(1,2) = net.optmloss;
            net.rl(1,3) = net.loss;
            net.rl(1,4) = net.loss;
        else
            net.rl(end+1, 1) = net.optmloss;
            net.rl(end, 2) = 0.99 * net.rl(end-1, 2) + 0.01 * net.optmloss;
            net.rl(end, 3) = net.loss;
            net.rl(end, 4) = 0.99 * net.rl(end-1, 4) + 0.01 * net.loss;
        end
    end
    fprintf('last batch optmloss:%.4f, loss:%.4f\n', net.optmloss, net.loss);
    toc;
    net.epoch = net.epoch + 1;
    if useSnapshot
        if (mod(epoch, opts.snapshotinterval)==0)
            save([net.name '_epoch', num2str(net.epoch)], 'net');
        end
    end
end
trainrslt = cnntest(net, opts, x, y);
cnnshowresult(trainrslt, opts, x, y);
if USETESTDATA
    testrslt = cnntest(net, opts, test_x, test_y);
    cnnshowresult(testrslt, opts, test_x, test_y);
end
if useSnapshot && mod(epoch, opts.snapshotinterval)~=0
    save([net.name '_epoch', num2str(net.epoch)], 'net');
end
end

