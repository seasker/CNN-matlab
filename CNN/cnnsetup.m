function net = cnnsetup(net,opts, x, y)
n = numel(net.layers);
net.regloss = 0;
net.epoch = 0;

for l = 1 : n  %  layer
    if strcmp(net.layers{l}.type, 'i')
        net.layers{l}.mapsize = size(squeeze(x(:, :,1, 1)));
        net.layers{l}.outputchannels = size(x,4);
        net.layers{l}.outputtype = 'a';
        rf = [1,1];
    elseif strcmp(net.layers{l}.type, 's')
        net.layers{l}.stride = [0, 0] + net.layers{l}.stride;
        net.layers{l}.poolsize = [0, 0] +  net.layers{l}.poolsize;
        net.layers{l}.mapsize = ceil((net.layers{l-1}.mapsize - net.layers{l}.poolsize) ./ net.layers{l}.stride) + 1;
        net.layers{l}.outputchannels = net.layers{l-1}.outputchannels;
        net.layers{l}.outputtype = 'a';
        rf = (rf-1).* net.layers{l}.stride + net.layers{l}.poolsize;  
    elseif strcmp(net.layers{l}.type, 'c')
        net.layers{l}.kernelsize = [0, 0] + net.layers{l}.kernelsize;
        net.layers{l}.stride = [0, 0] + net.layers{l}.stride;
        net.layers{l}.mapsize = ceil((net.layers{l-1}.mapsize - net.layers{l}.kernelsize) ./ net.layers{l}.stride) + 1;
        net.layers{l}.outputtype = 'a';
        if ~isfield(net.layers{l}, 'kernelfiller')
            net.layers{l}.kernelfiller = opts.paramfiller;
        end
        if ~isfield(net.layers{l}, 'droprate')
            net.layers{l}.droprate = opts.droprate;
        end
        if ~isfield(net.layers{l}, 'decay')
            net.layers{l}.decay = opts.decay;
        end
        rf = (rf -1).* net.layers{l}.stride + net.layers{l}.kernelsize;
        fan_out = net.layers{l}.outputchannels * prod(net.layers{l}.kernelsize);
        for j = 1 : net.layers{l}.outputchannels  %  output /
            fan_in = net.layers{l-1}.outputchannels * prod(net.layers{l}.kernelsize);
            for i = 1 : net.layers{l-1}.outputchannels  %  input map
                
                if strcmp(net.layers{l}.kernelfiller, 'xavier')
                    net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                elseif  strcmp(net.layers{l}.kernelfiller, 'gaussian')
                    net.layers{l}.k{i}{j} = 0.1 * randn(net.layers{l}.kernelsize);
                end
                net.layers{l}.ik{i}{j} = zeros(net.layers{l}.kernelsize);
                net.regloss = net.regloss + 0.5 * net.layers{l}.decay * sum(net.layers{l}.k{i}{j}(:).^2);
            end
            net.layers{l}.b{j} = 0;
            net.layers{l}.ib{j}= 0;
        end
    elseif strcmp(net.layers{l}.type, 'bn')
        net.layers{l}.mapsize = net.layers{l-1}.mapsize;
        net.layers{l}.outputchannels = net.layers{l-1}.outputchannels;
        if ~isfield(net.layers{l}, 'droprate')
            net.layers{l}.droprate = opts.droprate;
        end
        if ~isfiled(net.layers{l}, 'decay')
            net.layers{l}.decay = opts.decay;
        end
        if isfield(net.layers{l-1}, 'a')
            net.layers{l}.outputtype = 'a';
            for j = 1 : net.layers{l}.outputchannels
                net.layers{l}.gamma{j} = ones(net.layers{l}.mapsize)/100;
                net.layers{l}.igamma{j} = zeros(net.layers{l}.mapsize);
                net.layers{l}.beta{j} = 0;
                net.layers{l}.ibeta{j} = 0;
                net.regloss = net.regloss + 0.5 * net.layers{l}.decay * sum(net.layers{l}.gamma{j}(:).^2);
            end
        else
            net.layers{l}.outputtype = 'o';
            net.layers{l}.gamma = ones(net.layers{l}.outputchannels, 1)/100;
            net.layers{l}.igamma = zeros(net.layers{l}.outputchannels, 1);
            net.layers{l}.beta = zeros(net.layers{l}.outputchannels, 1);
            net.layers{l}.ibeta = zeros(net.layers{l}.outputchannels, 1);
            net.regloss = net.regloss + 0.5 * net.layers{l}.decay * sum(net.layers{l}.gamma(:).^2);
        end
    elseif strcmp(net.layers{l}.type, 'd')
        net.layers{l}.mapsize = net.layers{l-1}.mapsize;
        net.layers{l}.outputchannels = net.layers{l-1}.outputchannels;
        net.layers{l}.outputtype = net.layers{l-1}.outputtype;
        if ~isfield(net.layers{l}, 'droprate')
            net.layers{l}.droprate = opts.droprate;
        end
    elseif strcmp(net.layers{l}.type, 'fc')
        net.layers{l}.mapsize = [1,1];
        net.layers{l}.outputtype = 'o';
        if ~isfield(net.layers{l}, 'weightfiller')
            net.layers{l}.weightfiller = opts.paramfiller;
        end
        if ~isfield(net.layers{l}, 'droprate')
            net.layers{l}.droprate = opts.droprate;
        end
        if ~isfield(net.layers{l}, 'decay')
            net.layers{l}.decay = opts.decay;
        end
        inputchannels = prod(net.layers{l-1}.mapsize) * net.layers{l-1}.outputchannels;
        if l == n
            if isfield(net.layers{l},'outputchannels')
                assert(net.layers{l}.outputchannels == size(y, 1),...
                    'The last layer''s field outputchannels should equal class number, if you dont know, you can omit it');
            else
                net.layers{l}.outputchannels = size(y, 1);    
            end
        end
         outputchannels = net.layers{l}.outputchannels;
        if strcmp(net.layers{l}.weightfiller, 'xavier')
            net.layers{l}.w = (rand(outputchannels, inputchannels) - 0.5) * 2 * sqrt(6 / (inputchannels + outputchannels));
        elseif strcmp(net.layers{l}.weightfiller, 'gaussian')
            net.layers{l}.w = 0.1 * randn(outputchannels, inputchannels);
        end
        net.layers{l}.iw = zeros(outputchannels, inputchannels);
        net.layers{l}.b = zeros(outputchannels, 1);
        net.layers{l}.ib = zeros(outputchannels, 1);
        net.regloss = net.regloss + 0.5 * net.layers{l}.decay * sum(net.layers{l}.w(:).^2);
    end
end   
end
