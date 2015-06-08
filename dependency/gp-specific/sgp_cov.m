function [Cnoisy, dC_dz] = sgp_cov( GPmodel, z, Psi1 )

n=size(Psi1,2);

dsl = GPmodel.diagSqrtLambda .* exp(-z*GPmodel.idxbScaleEnabled);

% pair-wise distance
D = pdist( bsxfun( @times, Psi1.', dsl ) );

% covariance matrix without noise
Cv = GPmodel.normCov * exp(-0.5*(D.*D));
C = squareform(Cv); C(1:n+1:end) = GPmodel.normCov;

% covariance matrix with noise
Cnoisy = C + diag( repmat( GPmodel.noiseSigma2, 1, n ) );

if nargout>=2
    shl = GPmodel.diagSqrtLambda(GPmodel.idxbScaleEnabled);
    Dse = pdist( bsxfun( @times, Psi1(GPmodel.idxbScaleEnabled,:).', shl ) );
    dC_dz_v = exp(-2*z).*Cv.*(Dse.*Dse);
    dC_dz = squareform(dC_dz_v);
end

end
