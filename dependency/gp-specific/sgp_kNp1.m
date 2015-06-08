function [ kNp1, dkNp1_dpsiNp1 ]= sgp_kNp1( psiNp1, PsiN, GPmodel )

n=size(PsiN,2);

eta    = GPmodel.normCov;

D  = PsiN - repmat(psiNp1,1,n);
Ds = bsxfun(@times, D, vec(GPmodel.diagSqrtLambda));
kNp1 = eta * exp( -0.5*sum(Ds.*Ds,1) )';

if nargout>=2
    dkNp1_dpsiNp1 = bsxfun( @times, D.*repmat(kNp1.',size(D,1),1), vec(GPmodel.diagSqrtLambda).^2 );
end

end
