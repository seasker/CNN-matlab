function net = cnnapplygrads(net, opts)
n = numel(net.layers);
net.regloss = 0;
for l = 2 : n
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : net.layers{l}.outputchannels
            for i = 1 : net.layers{l-1}.outputchannels
                net.layers{l}.ik{i}{j}= opts.moment * net.layers{l}.ik{i}{j} + opts.alpha * net.layers{l}.dk{i}{j};
                net.layers{l}.k{i}{j} = net.layers{l}.k{i}{j} - net.layers{l}.ik{i}{j};
                net.regloss = net.regloss + 0.5 * net.layers{l}.decay * sum(net.layers{l}.k{i}{j}(:).^2);
            end
            net.layers{l}.ib{j} = opts.moment * net.layers{l}.ib{j} + opts.alpha * net.layers{l}.db{j};
            net.layers{l}.b{j} = net.layers{l}.b{j} - net.layers{l}.ib{j};
        end
    elseif strcmp(net.layers{l}.type, 'bn')
        if strcmp(net.layers{l}.outputtype, 'a')
            for j = 1 : net.layers{l}.outputchannels
                net.layers{l}.igamma{j} = opts.moment * net.layers{l}.igamma{j} + opts.alpha * net.layers{l}.dgamma{j};
                net.layers{l}.gamma{j} = net.layers{l}.gamma{j} - net.layers{l}.igamma{j};
                net.layers{l}.ibeta{j} = opts.moment * net.layers{l}.ibeta{j} + opts.alpha * net.layers{l}.dbeta{j};
                net.layers{l}.beta{j} = net.layers{l}.beta{j} - net.layers{l}.dbeta{j};
                net.regloss = net.regloss + 0.5 * net.layers{l}.decay * sum(net.layers{l}.gamma{j}(:).^2);
            end
        else
            net.layers{l}.igamma = opts.moment * net.layers{l}.igamma + opts.alpha * net.layers{l}.dgamma;
            net.layers{l}.gamma = net.layers{l}.gamma - net.layers{l}.igamma;
            net.layers{l}.ibeta = opts.moment * net.layers{l}.ibeta + opts.alpha * net.layers{l}.dbeta;
            net.layers{l}.beta = net.layers{l}.beta - net.layers{l}.dbeta;
            net.regloss = net.regloss + 0.5 * net.layers{l}.decay * sum(net.layers{l}.gamma(:).^2);
        end
    elseif strcmp(net.layers{l}.type, 'fc')
        net.layers{l}.iw = opts.moment * net.layers{l}.iw + opts.alpha * net.layers{l}.dw;
        net.layers{l}.w = net.layers{l}.w - net.layers{l}.iw;
        net.layers{l}.ib = opts.moment * net.layers{l}.ib + opts.alpha * net.layers{l}.db;
        net.layers{l}.b = net.layers{l}.b - net.layers{l}.ib;
        net.regloss = net.regloss + 0.5 * net.layers{l}.decay * sum(net.layers{l}.w(:).^2);
    end
end
end
