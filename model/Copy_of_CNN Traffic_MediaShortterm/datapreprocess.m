function [ train_x, train_y, test_x, test_y, datameta] = datapreprocess(origin_data, train_num, temporal_granu_x, temporal_granu_y, is_random)
assert(nargin >= 2, 'not enough arguments');
[space_dim,temporal_dim,channel_dim] = size(origin_data);
if nargin < 5, is_random = false; end;
if nargin < 4, temporal_granu_y = floor(space_dim/4); end;
if nargin < 3, temporal_granu_x = space_dim; end;
if is_random
    indices = randperm(1 : temporal_dim - temporal_granu_x -temporal_granu_y + 1);
else
    indices = 1 : temporal_dim - temporal_granu_x -temporal_granu_y + 1;
end
datameta.maxval = squeeze(max(max(origin_data)));
datameta.minval = squeeze(min(min(origin_data)));
datameta.meanval = squeeze(mean(mean(origin_data)));
datameta.channels = channel_dim;
train_x = zeros(space_dim,temporal_granu_x,train_num,channel_dim);
train_y = zeros(prod([space_dim,temporal_granu_y,channel_dim]),train_num);
for i = 1 : train_num
    xbegin = indices(i);
    xend = xbegin + temporal_granu_x - 1;
    ybegin = xend + 1;
    yend = ybegin + temporal_granu_y - 1;
    train_x(:,:,i,:) = origin_data(:,xbegin:xend,:);
    train_y(:,i) = reshape(origin_data(:,ybegin:yend,:),[],1);
end
test_num = temporal_dim -train_num -temporal_granu_x -temporal_granu_y + 1;
test_x = zeros(space_dim,temporal_granu_x,test_num,channel_dim);
test_y = zeros(prod([space_dim,temporal_granu_y,channel_dim]),test_num);
for i =1 : test_num
    xbegin = indices(i+train_num);
    xend = xbegin + temporal_granu_x - 1;
    ybegin = xend + 1;
    yend = ybegin + temporal_granu_y - 1;
    test_x(:,:,i,:) = origin_data(:,xbegin:xend,:);
    test_y(:,i) = reshape(origin_data(:,ybegin:yend,:),[],1);
end
end

