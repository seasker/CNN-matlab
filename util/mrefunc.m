function [ l ] = mrefunc(x, y)
valididx = find(y>0);
revec = abs((x(valididx)-y(valididx))./ (y(valididx)));
l = sum(revec)./ numel(revec);
end

