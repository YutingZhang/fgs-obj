function [ll, dll_dz] = sgp_negloglik( GPmodel, z, Psi1, f )
% USAGE: ll = sgp_loglik( GPmodel, X, Y )
% Specifically for GP with SEard covariance kernel, constant mean, and
%  Gaussian likelihood (noise)
% Psi is d*n matrix, Y is n*1 vector
% GPmodel is struct, whose fields are m0, diagSqrtLambda, normCov, noiseSigma2;


if nargout<=1
    % covariance matrix without noise
    KN = sgp_cov( GPmodel, z, Psi1 );
    % joint likelihood
    ll = sgp_negloglik_givenC( GPmodel, KN, f );
else
    % covariance matrix without noise
    [KN, dKN_dz] = sgp_cov( GPmodel, z, Psi1 );
    % joint likelihood
    [ll, dll_dKN] = sgp_negloglik_givenC( GPmodel, KN, f );    
    dll_dz = vec(dKN_dz).'*vec(dll_dKN);
end

end

