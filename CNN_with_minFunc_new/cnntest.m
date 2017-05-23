function [rslt] = cnntest(net, opts, x, y)
%  feedforward
net.phase = 'test';
n = numel(net.layers);
net = cnnff(net, opts, x, y);
rslt.type = net.type;
rslt.fv =  net.layers{n-1}.o;
rslt.pv =  net.layers{n}.o;
rslt.loss = net.loss;
if strcmp(net.type, 'R')
    rslt.mre = mrefunc(net.layers{n}.o, y);
end
end
