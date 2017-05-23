function net = cnntrainwithminfunc(net, opts,  x, y , test_x , test_y, minfuncopt )
global useBatchShuffle;
global useSnapshot;
if nargin > 4
    USETESTDATA = true;
end
net.phase = 'train';
m = size(x, 3); % m
numbatches = ceil(m / opts.batchsize);
net.rl = [];

for epoch = 1 : opts.numepochs
    if  rem(epoch-1, opts.testinterval)==0
        trainrslt = cnntest(net, opts, x, y);
        fprintf('Validation on TRAINDATA /epoch %d\n',net.epoch);
        cnnshowresult(trainrslt, opts, x, y);
        if USETESTDATA
            testrslt = cnntest(net, opts, test_x, test_y);
            fprintf('Validation on TESTDATA /epoch %d\n',net.epoch);
            cnnshowresult(testrslt, opts, test_x, test_y);
        end
    end
    fprintf('epoch %d/%d\n', epoch, opts.numepochs);
    tic;
    if useBatchShuffle
        order = randperm(m);
    else
        order = 1 : m;
    end
    for batch = 1 : numbatches
        batch_x = x(:, :, mod(order, numbatches) == batch-1, :);
        batch_y = y(:,    mod(order, numbatches) == batch-1, :);
        % Feedforward and backpropgation and update kernel
        index = 0;
        for l = 1 : numel(net.layers)
            if strcmp(net.layers{l}.type, 'c')
                for j = 1 : net.layers{l}.outputchannels
                    for i = 1 : net.layers{l-1}.outputchannels
                        theta(index+1:index+numel(net.layers{l}.k{i}{j})) = net.layers{l}.k{i}{j}(:);
                        index = index + numel(net.layers{l}.k{i}{j});
                    end
                    theta(index+1:index+numel(net.layers{l}.b{j})) = net.layers{l}.b{j}(:);
                    index = index + 1;
                end
            elseif strcmp(net.layers{l}.type, 'bn')
                if  strcmp(net.layers{l-1}.type, 'fc')
                    theta(index+1:index+numel(net.layers{l}.gamma)) = net.layers{l}.gamma(:);
                    index = index + numel(net.layers{l}.gamma);
                    theta(index+1:index+numel(net.layers{l}.beta)) = net.layers{l}.beta(:);
                    index = index + numel(net.layers{l}.beta);
                else
                    for j = 1 : net.layers{l}.outputchannels
                        theta(index+1:index+numel(net.layers{l}.gamma{j})) = net.layers{l}.gamma{j}(:);
                        index = index + numel(net.layers{l}.gamma{j});
                        theta(index+1:index+numel(net.layers{l}.beta{j})) = net.layers{l}.beta{j}(:);
                        index = index + numel(net.layers{l}.beta{j});
                    end
                end
            elseif strcmp(net.layers{l}.type, 'fc')
                theta(index+1:index+numel(net.layers{l}.w)) = net.layers{l}.w(:);
                index = index + numel(net.layers{l}.w);
                theta(index+1:index+numel(net.layers{l}.b)) = net.layers{l}.b(:);
                index = index + numel(net.layers{l}.b);
            end
        end
        theta = theta(:);
        [theta, bl] = minFunc(@cnnffloss,theta, minfuncopt, net, opts, batch_x, batch_y);
        net.regloss = 0;
        index = 0;
        for l = 1 : numel(net.layers)
            if strcmp(net.layers{l}.type, 'c')
                for j = 1 : net.layers{l}.outputchannels
                    for i = 1 : net.layers{l-1}.outputchannels
                        net.layers{l}.k{i}{j} = reshape(theta(index+1:index+numel(net.layers{l}.k{i}{j})),size(net.layers{l}.k{i}{j}));
                        index = index + numel(net.layers{l}.k{i}{j});
                        net.regloss = net.regloss + 0.5 * net.layers{l}.decay * sum(net.layers{l}.k{i}{j}(:).^2);
                    end
                    net.layers{l}.b{j} = reshape(theta(index+1:index+numel(net.layers{l}.b{j})),size(net.layers{l}.b{j}));
                    index = index + numel(net.layers{l}.b{j});
                end
            elseif strcmp(net.layers{l}.type, 'bn')
                if isfield(net.layers{l-1},'a')
                    for j = 1 : net.layers{l}.outputchannels
                        net.layers{l}.gamma{j} = reshapae(theta(index+1:index+numel(net.layers{l}.gamma{j})),size(net.layers{l}.gamma{j}));
                        index = index + numel(net.layers{l}.gamma{j});
                        net.layers{l}.beta{j} = reshape(theta(index+1:index+numel(net.layers{l}.beta{j})),size(net.layers{l}.beta{j}));
                        index = index + numel(net.layers{l}.beta{j});
                        net.regloss = net.regloss + 0.5 * net.layers{l}.decay * sum(net.layers{l}.gamma{j}(:).^2);
                    end
                    
                else
                    net.layers{l}.gamma = reshape(theta(index+1:index+numel(net.layers{l}.gamma)),size(net.layers{l}.gamma));
                    index = index + numel(net.layers{l}.gamma);
                    net.layers{l}.beta = reshape(theta(index+1:index+numel(net.layers{l}.beta)), size(net.layers{l}.beta));
                    index = index + numel(net.layers{l}.beta);
                    net.regloss = net.regloss + 0.5 * net.layers{l}.decay * sum(net.layers{l}.gamma(:).^2);
                end
            elseif strcmp(net.layers{l}.type, 'fc')
                net.layers{l}.w = reshape(theta(index+1:index+numel(net.layers{l}.w)),size(net.layers{l}.w));
                index = index + numel(net.layers{l}.w);
                net.layers{l}.b = reshape(theta(index+1:index+numel(net.layers{l}.b)),size(net.layers{l}.b));
                index = index + numel(net.layers{l}.b);
                net.regloss = net.regloss + 0.5 * net.layers{l}.decay * sum(net.layers{l}.w(:).^2);
            end
        end
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
