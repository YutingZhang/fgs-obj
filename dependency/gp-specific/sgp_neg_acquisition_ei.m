function [a, da_psiNp1] = sgp_neg_acquisition_ei(GPmodel,psiNp1,PsiN,fN,fN_hat,KN)

if ~exist('KN','var') || isempty(KN)
    KN = sgp_KN( PsiN, GPmodel );
end

% psiNp1 = reshape(psiNp1,numel(psiNp1),1);

if nargout<=1

    kNp1 = sgp_kNp1( psiNp1, PsiN, GPmodel );
    mu   = sgp_posterior_mu( kNp1, KN, fN, GPmodel );
    s2   = sgp_posterior_s2( kNp1, KN, GPmodel );
    a    = sgp_ei( mu, s2, fN_hat );
    a    = -a;

else
    
    [kNp1, dkNp1_dpsiNp1] = sgp_kNp1( psiNp1, PsiN, GPmodel );
    [mu, dmu_dNp1] = sgp_posterior_mu( kNp1, KN, fN, GPmodel );
    [s2, ds2_dNp1] = sgp_posterior_s2( kNp1, KN, GPmodel );
    [a, da_dmu, da_ds2] = sgp_ei( mu, s2, fN_hat );

    da_psiNp1 = reshape(dkNp1_dpsiNp1, numel(psiNp1), numel(kNp1) ) * ...
        ( vec(dmu_dNp1)*da_dmu + ...
        vec(ds2_dNp1)*da_ds2 );
    
    a = -a; da_psiNp1 = -da_psiNp1;

end

end
