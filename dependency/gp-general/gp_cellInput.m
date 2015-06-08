function [nlZ,dnlZ] = gp_cellInput( hyp_, inf_, mean_, cov_, lik_, X_, Y_ )

n = length(X_);
assert(n==length(Y_),'X_ should have the same length as Y_');
A = nan(n,1);
B = cell(n,1);
for i = 1:n
    [A(i), B{i}] = gp(hyp_,inf_,mean_,cov_,lik_,double(X_{i}),double(Y_{i}));
end
nlZ = mean(A);
B = cell2mat(B);

dnlZ=B(1);
F = fieldnames(B);
for k=1:length(F)
    eval(sprintf('catDim = length(size(B(1).%s))+1;', F{k}));
    eval(sprintf('dnlZ.%s = mean( cat(catDim,B.%s), catDim );',F{k},F{k}));
end

end


