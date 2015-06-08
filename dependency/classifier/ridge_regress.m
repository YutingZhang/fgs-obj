function [x, losses] = ridge_regress(A, y, lambda, method)
% solve for x in min_x ||Ax - y||^2 + lambda*||x||^2

if ~exist('method','var') || isempty(method)
    method = 'ridge_reg_chol';
end

switch method
case 'ridge_reg_chol'
  % solve for x in min_x ||Ax - y||^2 + lambda*||x||^2
  %
  % solve (A'A + lambdaI)x = A'y for x using cholesky factorization
  % R'R = (A'A + lambdaI)
  % R'z = A'y  :  solve for z  =>  R'Rx = R'z  =>  Rx = z
  % Rx = z     :  solve for x
  R = chol(A'*A + lambda*eye(size(A,2)));
  z = R' \ (A'*y);
  x = R \ z;
case 'ridge_reg_inv'
  % solve for x in min_x ||Ax - y||^2 + lambda*||x||^2
  x = inv(A'*A + lambda*eye(size(A,2)))*A'*y;
case 'ls_mldivide'
  % solve for x in min_x ||Ax - y||^2
  if lambda > 0
    warning('ignoring lambda; no regularization used');
  end
  x = A\y;
end
losses = 0.5 * (A*x - y).^2;
