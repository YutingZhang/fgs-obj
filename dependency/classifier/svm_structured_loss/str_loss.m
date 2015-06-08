function y = str_loss(x, lb, ub)

if ~exist('lb', 'var'),
    lb = 0;
end
if ~exist('ub', 'var'),
    ub = 1;
end

y = -(x-ub)/(ub-lb);
y = min(max(y, 0), 1);

return;
