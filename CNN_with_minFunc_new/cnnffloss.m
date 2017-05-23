function [functionval, derivationval] = cnnffloss( theta , net, opts, data_x, data_y)
index = 0;
for l = 1 : numel(net.layers)
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : net.layers{l}.outputchannels
            for i = 1 : net.layers{l-1}.outputchannels
                net.layers{l}.k{i}{j} = reshape(theta(index+1:index+numel(net.layers{l}.k{i}{j})),size(net.layers{l}.k{i}{j}));
                index = index + numel(net.layers{l}.k{i}{j});
            end
            net.layers{l}.b{j} = reshape(theta(index+1:index+numel(net.layers{l}.b{j})),size(net.layers{l}.b{j}));
            index = index + numel(net.layers{l}.b{j});
        end
    elseif strcmp(net.layers{l}.type, 'bn')
        if isfield(net.layers{l-1},'a')
            for j = 1 : net.layers{l}.outputchannels
                net.layers{l}.gamma{j} = reshape(theta(index+1:index+numel(net.layers{l}.gamma{j})),size(net.layers{l}.gamma{j}));
                index = index + numel(net.layers{l}.gamma{j});
                net.layers{l}.beta{j} = reshape(theta(index+1:index+numel(net.layers{l}.beta{j})),size(net.layers{l}.beta{j}));
                index = index + numel(net.layers{l}.beta{j});
            end
        else
            net.layers{l}.gamma = reshape(theta(index+1:index+numel(net.layers{l}.gamma)),size(net.layers{l}.gamma));
            index = index + numel(net.layers{l}.gamma);
            net.layers{l}.beta = reshape(theta(index+1:index+numel(net.layers{l}.bata)),size(net.layers{l}.beta));
            index = index + numel(net.layers{l}.beta);
        end
    elseif strcmp(net.layers{l}.type, 'fc')
        net.layers{l}.w = reshape(theta(index+1:index+numel(net.layers{l}.w)),size(net.layers{l}.w));
        index = index + numel(net.layers{l}.w);
        net.layers{l}.b = reshape(theta(index+1:index+numel(net.layers{l}.b)),size(net.layers{l}.b));
        index = index + numel(net.layers{l}.b);
    end
end
net = cnnff(net,opts, data_x, data_y);
net = cnnbp(net);
%compute function value and gradient with vector converted
functionval = net.optmloss;
index = 0;
for l = 1 : numel(net.layers)
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : net.layers{l}.outputchannels
            for i = 1 : net.layers{l-1}.outputchannels
                derivationval(index+1:index+numel(net.layers{l}.k{i}{j})) = net.layers{l}.dk{i}{j}(:);
                index = index + numel(net.layers{l}.k{i}{j});
            end
            derivationval(index+1:index+numel(net.layers{l}.b{j})) = net.layers{l}.db{j};
            index = index + numel(net.layers{l}.b{j});
        end
    elseif strcmp(net.layers{l}.type, 'bn')
        if isfield(net.layers{l-1},'a')
            for j = 1 : net.layers{l}.outputchannels
                derivationval(index+1:index+numel(net.layers{l}.gamma{j})) = net.layers{l}.dgamma{j};
                index = index + numel(net.layers{l}.gamma{j});
                derivationval(index+1:index+numel(net.layers{l}.beta{j})) = net.layers{l}.dbeta{j};
                index = index + numel(net.layers{l}.beta{j});
            end
            
        else
            derivationval(index+1:index+numel(net.layers{l}.gamma)) = net.layers{l}.dgamma;
            index = index + numel(net.layers{l}.gamma);
            derivationval(index+1:index+numel(net.layers{l}.beta)) = net.layers{l}.dbeta;
            index = index + numel(net.layers{l}.beta);
        end
    elseif strcmp(net.layers{l}.type, 'fc')
        derivationval(index+1:index+numel(net.layers{l}.w)) = net.layers{l}.dw;
        index = index + numel(net.layers{l}.w);
        derivationval(index+1:index+numel(net.layers{l}.b)) = net.layers{l}.db;
        index = index + numel(net.layers{l}.b);
    end
end
derivationval= derivationval(:);
end

