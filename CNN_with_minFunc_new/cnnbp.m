function net = cnnbp(net)
n = numel(net.layers);
batchsize = size(net.layers{n}.o, 2);

%%  backprop deltas
for l = n : -1 : 1
    if strcmp(net.layers{l}.type, 'fc')
        if l < n
            net.layers{l}.od = gradfunc(net.layers{l}.od, net.layers{l}.o, net.layers{l}.activefunc);
        end
        net.layers{l-1}.od = net.layers{l}.w' * net.layers{l}.od;
    elseif strcmp(net.layers{l}.type, 'c')
        if isfield(net.layers{l}, 'od')
            for j = 1 : net.layers{l}.outputchannels
                net.layers{l}.ad{j} = net.layers{l}.od((j-1) * prod(net.layers{l}.mapsize) + 1: j* prod(net.layers{l}.mapsize), :);
                net.layers{l}.ad{j} = reshape(net.layers{l}.ad{j}, [net.layers{l}.mapsize,batchsize]);
            end
        end
        if isfield(net.layers{l},'activefunc')
            for j = 1 : net.layers{l}.outputchannels
                net.layers{l}.ad{j} = gradfunc( net.layers{l}.ad{j}, net.layers{l}.a{j}, net.layers{l}.activefunc);
            end
        end
        for j = 1 : net.layers{l}.outputchannels
            net.layers{l}.zd{j} = zeros([(net.layers{l}.mapsize - 1) .* net.layers{l}.stride + 1, batchsize]);
            net.layers{l}.zd{j}(1:net.layers{l}.stride(1):end, 1:net.layers{l}.stride(2):end, :) = net.layers{l}.ad{j};
        end
        for i = 1 : net.layers{l-1}.outputchannels
            d = zeros(size(net.layers{l-1}.pada{i}));
            for j = 1 :net.layers{l}.outputchannels
                
                d = d + convn(net.layers{l}.zd{j}, flipall(net.layers{l}.k{i}{j}), 'full');
            end
            net.layers{l-1}.ad{i} = padrmv(d, net.layers{l-1}.padsz);
        end
    elseif strcmp(net.layers{l}.type, 's')
        if isfield(net.layers{l},'od')
            for j = 1 : net.layers{l}.outputchannels
                net.layers{l}.ad{j} = net.layers{l}.od((j-1)*prod(net.layers{l}.mapsize)+1:j*prod(net.layers{l}.mapsize), :);
                net.layers{l}.ad{j} = reshape(net.layers{l}.ad{j}, [net.layers{l}.mapsize,batchsize]);
            end
        end
        if isfield(net.layers{l}, 'activefunc')
            for j = 1 : net.layers{l}.outputchannels
                net.layers{l}.ad{j} = gradfunc(net.layers{l}.ad{j}, net.layers{l}.a{j}, net.layers{l}.activefunc);
            end
        end
        if strcmp(net.layers{l}.poolmethod, 'a')
            for i = 1 : net.layers{l-1}.outputchannels
                zd = zeros([(net.layers{l}.mapsize - 1) .* net.layers{l}.stride + 1, batchsize]);
                zd(1:net.layers{l}.stride(1):end, 1:net.layers{l}.stride(2):end, :) = net.layers{l}.ad{i};
                d = convn(zd, flipall(ones(net.layers{l}.poolsize)/prod(net.layers{l}.poolsize)), 'full');
                net.layers{l-1}.ad{i} = padrmv(d, net.layers{l-1}.padsz);
            end
        elseif strcmp(net.layers{l}.poolmethod, 'm') || strcmp(net.layers{l}.poolmethod, 's')
            for i = 1 : net.layers{l-1}.outputchannels
                d = zeros(size(net.layers{l-1}.pada{i}));
                d(net.layers{l}.maxpos{i}) = d(net.layers{l}.maxpos{i}) + net.layers{l}.ad{i}(:);
                net.layers{l-1}.ad{i} = padrmv(d, net.layers{l-1}.padsz);
            end
        end
    elseif strcmp(net.layers{l}.type, 'bn')
        if isfield(net.layers{l}, 'a')
            if isfield(net.layers{l}, 'od')
                for j = 1 : net.layers{l}.outputchannels
                    net.layers{l}.ad{j} = net.layers{l}.od((j-1)*prod(net.layers{l}.mapsize)+1:j*prod(net.layers{l}.mapsize), :);
                    net.layers{l}.ad{j} = reshape(net.layers{l}.ad{j}, [net.layers{l}.mapsize,batchsize]);
                end
            end
            if isfield(net.layers{l}, 'activefunc')
                for j = 1 : net.layers{l}.outputchannels
                    net.layers{l}.ad{j} = gradfunc(net.layers{l}.ad{j}, net.layers{l}.a{j}, net.layers{l}.activefunc);
                end
            end
            for i = 1 : net.layers{l-1}.outputchannels
                nd = bsxfun(@times, net.layers{l}.ad{i}, net.layers{l}.gamma{i});
                dev = bsxfun(@minus, net.layers{l-1}.a{i} , net.layers{l-1}.mu{i});
                invsigma = 1 ./ sqrt(net.layers{l-1}.sigma{i}.^2 + eps) ;
                sigma2d = sum(-0.5 * bsxfun(@times, nd .* net.layers{l-1}.na{i}, invsigma.^2), 3);
                mud = sum(bsxfun(@times, nd, -invsigma), 3);
                net.layers{l-1}.ad{i} = bsxfun(@times, nd, invsigma) + bsxfun(@plus, bsxfun(@times, 2*dev, sigma2d), mud) / batchsize;
            end
        else
            net.layers{l}.od = gradfunc(net.layers{l}.od, net.layers{l}.o, net.layers{l}.activefunc);
            nd = bsxfun(@times, net.layers{l}.od, net.layers{l}.gamma);
            dev = bsxfun(@minus, net.layers{l-1}.o, net.layers{l-1}.mu);
            invsigma = 1./ sqrt(net.layers{l-1}.sigma.^2+eps);
            sigma2d = sum(-0.5 * bsxfun(@times, nd .* net.layers{l-1}.no,invsigma.^2), 2);
            mud = sum(bsxfun(@times, nd, -invsigma), 2);
            net.layers{l-1}.od = bsxfun(@times, nd, invsigma) + bsxfun(@plus, bsxfun(@times, 2*dev, sigma2d), mud)/ batchsize;
        end
    elseif strcmp(net.layers{l}.type, 'd')
        if isfield(net.layers{l}, 'a')
            if isfield(net.layers{l}, 'od')
                for j = 1 : net.layers{l}.outputchannels
                    net.layers{l}.ad{j} = net.layers{l}.od((j-1)*prod(net.layers{l}.mapsize)+1:j*prod(net.layers{l}.mapsize), :);
                    net.layers{l}.ad{j} = reshape(net.layers{l}.ad{j}, [net.layers{l}.mapsize,batchsize]);
                end
            end
            for i = 1 : net.layers{l}.outputchannels
                net.layers{l-1}.ad{i} = net.layers{l}.ad{i} .* net.layers{l}.dropmask{i};
            end
        else
            net.layers{l-1}.od = net.layers{l}.od .* net.layers{l}.dropmask; 
        end
    end
end
%%  calc gradients
for l = 2 : n
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : net.layers{l}.outputchannels
            for i = 1 : net.layers{l-1}.outputchannels
                net.layers{l}.dk{i}{j} = (convn(flipall(net.layers{l - 1}.pada{i}), net.layers{l}.zd{j}, 'valid') + ...
                    net.layers{l}.decay * net.layers{l}.k{i}{j}) / batchsize;
            end
            net.layers{l}.db{j} = sum(net.layers{l}.zd{j}(:)) / batchsize;%
        end
    elseif strcmp(net.layers{l}.type, 'bn')
        if strcmp(net.layers{l}, 'a')
            for j = 1 : net.layers{l}.outputchannels
                net.layers{l}.dgamma{j} = (sum(net.layers{l}.ad{j} .* net.layers{l}.a{j}, 3) + ...
                net.layers{l}.decay * net.layers{l}.gamma{j})/batchsize;
                net.layers{l}.dbeta{j} = sum(net.layers{l}.a{j}(:))/batchsize;
            end
        else
            net.layers{l}.dgamma = (sum(net.layers{l}.od .* net.layers{l}.o, 2)+ net.layers{l}.decay * net.layers{l}.gamma) / batchsize;
            net.layers{l}.dbeta = sum(net.layers{l}.od, 2)/batchsize;   
        end
    elseif strcmp(net.layers{l}.type, 'fc')
        net.layers{l}.dw = (net.layers{l}.od * net.layers{l-1}.o' + net.layers{l}.decay * net.layers{l}.w) / batchsize;
        net.layers{l}.db = mean(net.layers{l}.od, 2);
    end
end

end
