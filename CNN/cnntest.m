function [rslt] = cnntest(net, opts, x, y)
%  feedforward
net.phase = 'test';
n = numel(net.layers);
if ~isfield(opts, 'testbatchsize')
    opts.testbatchsize = opts.batchsize;
end
m = size(x, 3); % m
numbatches = double(floor(m / opts.batchsize));
rslt.type = net.type;
rslt.fv =  [];
rslt.pv =  [];
rslt.sloss = 0;
for batch = 1 : numbatches
    batch_x = x(:, :, (batch-1)*opts.testbatchsize+1:batch*opts.testbatchsize, :);
    batch_y = y(:,    (batch-1)*opts.testbatchsize+1:batch*opts.testbatchsize);
    net = cnnff(net, opts, batch_x, batch_y);
    rslt.fv = [rslt.fv,net.layers{n-1}.o];
    rslt.pv = [rslt.pv,net.layers{n}.o];
    rslt.sloss = rslt.sloss + net.loss * size(batch_x, 3);
end
if mod(m, opts.batchsize) ~= 0
    batch_x = x(:, :, numbatches*opts.testbatchsize+1:end, :);
    batch_y = y(:,    numbatches*opts.testbatchsize+1:end);
    net = cnnff(net, opts, batch_x, batch_y);
    rslt.fv = [rslt.fv,net.layers{n-1}.o];
    rslt.pv = [rslt.pv,net.layers{n}.o];
    rslt.sloss = rslt.sloss + net.loss * size(batch_x, 3);
end
rslt.loss = rslt.sloss / m;
if strcmp(net.type, 'R')
    rslt.mre = mrefunc(rslt.pv, y);
end
end
