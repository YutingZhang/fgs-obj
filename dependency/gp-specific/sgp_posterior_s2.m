function [s2, ds2_dkNp1] = sgp_posterior_s2( kNp1, KN, GPmodel )

eta = GPmodel.normCov;

t = (KN\kNp1);
s2 = eta+GPmodel.noiseSigma2 - kNp1'*t;
ds2_dkNp1 = -2 * t;

end
