function [l, g] = hmsefunc(x, y)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明

e = x - y;
l =  0.5 * sum(e(:).^2) / size(x, 2);
g =  e .* (x .* (1 - x)); 
end

