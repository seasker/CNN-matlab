function net = cnnff(net, opts, x, y)
n = numel(net.layers);
batchsize = size(x,3);
%% h*w*n
for l = 1 : n   %  for each layer
    if strcmp(net.layers{l}.type, 'i')
        for j = 1 : net.layers{l}.outputchannels
            net.layers{l}.a{j} = x(:,:,:,j);
        end
    elseif strcmp(net.layers{l}.type, 'c')
        %  !!below can probably be handled by insane matrix operations
        net.layers{l-1}.padsz = (net.layers{l}.mapsize - 1) .* net.layers{l}.stride + net.layers{l}.kernelsize - net.layers{l-1}.mapsize;
        for j = 1 : net.layers{l}.outputchannels   %  for each output map
            
            z = zeros([(net.layers{l}.mapsize - 1) .* net.layers{l}.stride + 1, batchsize]);
            for i = 1 : net.layers{l-1}.outputchannels   %  for each input map
                net.layers{l-1}.pada{i} = padadd(net.layers{l-1}.a{i}, net.layers{l-1}.padsz);
                z = z + convn(net.layers{l - 1}.pada{i}, net.layers{l}.k{i}{j}, 'valid');
            end
            %  add bias, pass through nonlinearity
            z = z + net.layers{l}.b{j};
            net.layers{l}.a{j} = z(1:net.layers{l}.stride(1):end, 1:net.layers{l}.stride(2):end, :);
        end
        if isfield(net.layers{l}, 'activefunc')
            for j = 1 : net.layers{l}.outputchannels
                net.layers{l}.a{j} = net.layers{l}.activefunc(net.layers{l}.a{j});
            end
        end
        
    elseif strcmp(net.layers{l}.type, 's')
        %  downsample
        net.layers{l-1}.padsz = (net.layers{l}.mapsize - 1) .* net.layers{l}.stride + net.layers{l}.poolsize - net.layers{l-1}.mapsize;
        if strcmp(net.layers{l}.poolmethod, 'a')
            for j = 1 : net.layers{l}.outputchannels
                net.layers{l-1}.pada{j} = padadd(net.layers{l-1}.a{j}, net.layers{l-1}.padsz);
                z = convn(net.layers{l - 1}.pada{j}, ones(net.layers{l}.poolsize) / (prod(net.layers{l}.poolsize)), 'valid');
                net.layers{l}.a{j} = z(1:net.layers{l}.stride(1):end, 1:net.layers{l}.stride(2):end, :);
            end
        elseif strcmp(net.layers{l}.poolmethod, 'm')
            for j = 1 : net.layers{l}.outputchannels
                net.layers{l-1}.pada{j} = padadd(net.layers{l-1}.a{j}, net.layers{l-1}.padsz);
                [net.layers{l}.a{j}, net.layers{l}.maxpos{j}] = MaxPooling(net.layers{l - 1}.pada{j}, net.layers{l}.poolsize ,net.layers{l}.stride);
            end
        elseif strcmp(net.layers{l}.poolmethod, 's')
            if strcmp(net.phase, 'test')
                for j = 1 : net.layers{l}.outputchannels
                    net.layers{l-1}.pada{j} = padadd(net.layers{l-1}.a{j}, net.layers{l-1}.padsz);
                    z = convn(net.layers{l - 1}.pada{j}.^2, ones(net.layers{l}.poolsize), 'valid')./convn(net.layers{l - 1}.pada{j}, ones(net.layers{l}.poolsize), 'valid');
                    net.layers{l}.a{j} =z(1:net.layers{l}.stride(1):end, 1:net.layers{l}.stride(2):end, :);
                end
            else
                for j = 1 : net.layers{l}.outputchannels
                    net.layers{l-1}.pada{j} = padadd(net.layers{l-1}.a{j}, net.layers{l-1}.padsz);
                    [net.layers{l}.a{j}, net.layers{l}.maxpos{j}] = StochasticPooling(net.layers{l - 1}.pada{j},net.layers{l}.poolsize,net.layers{l}.stride); 
                end
            end
        end
        if isfield(net.layers{l}, 'activefunc')
            for j = 1 : net.layers{l}.outputchannels
                net.layers{l}.a{j} = net.layers{l}.activefunc(net.layers{l}.a{j});
            end
        end
    elseif strcmp(net.layers{l}.type, 'bn')
        if strcmp(net.phase,'test') && isfield(net.layers{l}, 'mu') && isfield(net.layers{l},'sigma')
            if strcmp(net.layer{l}.outputtype, 'a')
                for j = 1 : net.layers{l}.outputchannels
                    net.layers{l-1}.na{j} = bsxfun(@rdevide, bsxfun(@minus, net.layers{l-1}.a{j}, net.layers{l-1}.mu{j}), sqrt(net.layers{l-1}.sigma{j}.^2+eps));
                    net.layers{l}.a{j} = net.layers{l}.activefunc(bsxfun(@time, net.layers{l-1}.na{j}, net.layers{l}.gamma{j}) + net.layers{l}.beta{j});
                end
            else
                net.layers{l}.no = bsxfun(@rdevide, bsxfun(@minus, net.layers{l-1}.o, net.layers{l-1}.mu), sqrt(net.layers{l-1}.sigma.^2+eps));
                net.layers{l}.o = net.layers{l}.activefunc(bsxfun(@plus, bsxfun(@time, net.layers{l-1}.no, net.layers{l}.gamma), net.layers{l}.beta));
            end
        else
            if strcmp(net.layers{l}.outputtype, 'a')
                for j = 1 : net.layers{l}.outputchannels
                    net.layers{l-1}.mu{j} = mean(net.layers{l-1}.a{j}, 3);
                    net.layers{l-1}.sigma{j} = mean(bsxfun(@minus, net.layers{l-1}.a{j}, net.layers{l-1}.mu{j}).^2, 3);
                    net.layers{l-1}.na{j} = bsxfun(@rdivide, bsxfun(@minus, net.layers{l-1}.a{j}, net.layers{l-1}.mu{j}), sqrt(net.layers{l-1}.sigma{j}.^2+eps));
                    net.layers{l}.a{j} = bsxfun(@times, net.layers{l-1}.na{j}, net.layers{l}.gamma{j}) + net.layers{l}.beta{j};
                end
                if isfield(net.layers{l}, 'activefunc')
                    for j = 1 : net.layers{l}.outputchannels
                        net.layers{l}.a{j} = net.layers{l}.activefunc(net.layers{l}.a{j});
                    end
                end
            else
                net.layers{l-1}.mu = mean(net.layers{l-1}.o, 2);
                net.layers{l-1}.sigma = mean(bsxfun(@minus, net.layers{l-1}.o, net.layers{l-1}.mu).^2, 2);
                net.layers{l-1}.no = bsxfun(@rdivide ,bsxfun(@minus, net.layers{l-1}.o, net.layers{l-1}.mu), sqrt(net.layers{l-1}.sigma.^2+eps));
                net.layers{l}.o = bsxfun(@plus, bsxfun(@times, net.layers{l-1}.no, net.layers{l}.gamma), net.layers{l}.beta);
                if isfield(net.layers{l},'activefunc')
                    net.layers{l}.o = net.layers{l}.activefunc(net.layers{l}.o);
                end
            end
        end
    elseif strcmp(net.layers{l}.type, 'd')
        if strcmp(net.layers{l}.outputtype, 'a')
            for j = 1 : net.layers{l}.outputchannels
                net.layers{l}.dropmask{j} = rand(net.layers{l-1}.a{j})>net.layers{l}.droprate;
                net.layers{l}.a{j} = net.layers{l-1}.a{j} .* net.layers{l}.dropmask{j};
            end
        else
            net.layers{l}.dropmask = rand(net.layers{l-1}.o) > net.layers{l}.droprate;
            net.layers{l}.o = net.layers{l-1}.o .* net.layers{l}.dropmask;
        end
    elseif strcmp(net.layers{l}.type, 'fc')
        if  strcmp(net.layers{l-1}.outputtype, 'a')
            net.layers{l-1}.o = [];
            for i = 1 : net.layers{l-1}.outputchannels
                net.layers{l-1}.o = [net.layers{l-1}.o;reshape(net.layers{l-1}.a{i},prod(net.layers{l-1}.mapsize),batchsize)];
            end
        end
        net.layers{l}.o = bsxfun(@plus, net.layers{l}.w * net.layers{l-1}.o, net.layers{l}.b);
        if isfield(net.layers{l},'activefunc')
            net.layers{l}.o = net.layers{l}.activefunc(net.layers{l}.o);
        end
    end
end
[net.loss, net.layers{n}.od] = opts.lossfunc(net.layers{n}.o, y);
net.optmloss = net.loss + net.regloss / batchsize ;
end
