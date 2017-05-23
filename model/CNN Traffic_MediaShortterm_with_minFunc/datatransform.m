function [trans_x, trans_y] = datatransform(origin_x, origin_y, originmeta, transmeta)
m = size(origin_x, 3);
channels = size(origin_x, 4);
trans_x = (origin_x - reshape(originmeta.minval,1,1,1,[]))./ reshape((originmeta.maxval - originmeta.minval),1,1,1,[]) .* ...
    reshape((transmeta.maxval - transmeta.minval),1,1,1,[]) + reshape(transmeta.minval,1,1,1,[]);
trans_y = (reshape(origin_y,[],channels,m) - reshape(originmeta.minval,1,[])) ./ reshape((originmeta.maxval - originmeta.minval),1,[]) .* ...
    reshape((transmeta.maxval - transmeta.minval),1,[]) + reshape(transmeta.minval,1,[]);
trans_y = reshape(trans_y,[],m);
end

