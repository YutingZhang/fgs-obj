function [mu, dmu_dkNp1] = sgp_posterior_mu( kNp1, KN, fN, GPmodel )

% mu = m0 + kNp1'*inv(KN)*(f-m0);
dmu_dkNp1 = (KN\(fN-GPmodel.m0));
mu = GPmodel.m0 + kNp1'*dmu_dkNp1;

