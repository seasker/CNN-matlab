function [l, g] = msrefunc(x, y)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
e = x - y;
re = zeros(size(y));
re(y>0) = abs(e(y>0)./ y(y>0));
l = (1/2 * sum(e(:).^2) + sum(re(:)))/ size(x, 2);
gradre = zeros(size(y));
gradre(y>0) = 1 ./ abs(y(y>0)).* sign(x(y>0)-y(y>0));
g = (e + gradre) .* (x .* (1 - x));

end