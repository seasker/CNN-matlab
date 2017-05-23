function [detaout] = gradfunc(detain, funcval, funchandle)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
if isequal(funchandle, @sigm)
    detaout = funcval.*(1-funcval).* detain;
elseif isequal(funchandle, @tanh)
    detaout = (1 - funcval.^2) .* detain;
elseif isequal(funchandle, @relu)
    detaout = double(funcval>0) .* detain;
elseif isequal(funchandle, @softplus)
    detaout = (1-exp(-funcval)) .* detain;
elseif isequal(funchandle, @linearfun)
    detaout = ones(size(funcval)) .* detain;
elseif isequal(funchandle, @tanh_opt)
   
end
end

