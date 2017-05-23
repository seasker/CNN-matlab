function [ x ] = padrmv( padx, szpad)
%UNTITLED Summary of this function goes here
% Pad with the boundary of the two-dimensional matrix
szpadh = szpad(1);
szpadw = szpad(2);
x = padx(floor(szpadh/2)+1:end-ceil(szpadh/2), floor(szpadw/2)+1:end-ceil(szpadw/2), :);
end

