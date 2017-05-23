function [l, g ] = softcefunc(x, y )
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
e = -y .* log(x);
l = sum(e(:)) / size(x, 2);
g = x - y;
end

