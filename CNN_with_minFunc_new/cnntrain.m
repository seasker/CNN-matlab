function net = cnntrain(net, opts, x, y, test_x , test_y)
global useBatchShuffle;
global useSnapshot;
if nargin > 4
    USETESTDATA = true;
end
net.phase = 'train';
m = size(x, 3); % m
numbatches = ceil(m / double(opts.batchsize));

net.rl = [];
for epoch = 1 : opts.numepochs
    if rem(epoch-1, opts.testinterval)==0
        trainrslt = cnntest(net, opts, x, y);
        fprintf('----------------Validation on TRAINDATA /epoch %d------------------\n',net.epoch);
        cnnshowresult(trainrslt, opts, x, y);
        if USETESTDATA
            testrslt = cnntest(net, opts, test_x, test_y);
            fprintf('----------------Validation on TESTDATA /epoch %d------------------\n',net.epoch);
            cnnshowresult(testrslt, opts, test_x, test_y);
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
        batch_x = x(:, :, mod(order, numbatches) == batch-1, :);
        batch_y = y(:,    mod(order, numbatches) == batch-1,:);
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
            save(['cnn_epoch', num2str(net.epoch)],'net');
        end
    end
end
if useSnapshot && mod(epoch, opts.snapshotinterval)~=0
    save(['cnn_epoch', num2str(net.epoch)],'net');
end
end

