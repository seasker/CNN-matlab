function y = softmax(x)
    % Softmax function
    % mu(i,c) = exp(eta(i,c))/sum_c' exp(eta(i,c'))

    % This file is from matlabtools.googlecode.com
    c = 1;
    tmp = exp(c*x);
    z = sum(tmp, nargin);
    y = bsxfun(@rdivide, tmp, z);
end