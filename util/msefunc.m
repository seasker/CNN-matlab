function [l, g] = msefunc(x, y, coefficient)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
if nargin < 3, coefficient = 0.5; end

e = x - y;
l =  coefficient * sum(e(:).^2) / size(x, 2);
g = 2* coefficient *e .* (x .* (1 - x)); 
end

