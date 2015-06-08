function GPmodel = sgp_model_from_general( hyp )

GPmodel.m0 = hyp.mean;
GPmodel.diagSqrtLambda = 1./exp( vec(hyp.cov(1:4)).' );
GPmodel.normCov    = exp( 2*hyp.cov(5) );
GPmodel.noiseSigma2 = exp( 2*hyp.lik );
GPmodel.idxbScaleEnabled = boolean([1 1 0 0]);

end
