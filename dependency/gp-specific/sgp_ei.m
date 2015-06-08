function [a, dadm, dads2] = sgp_ei( mu, s2, fN_hat )

s  = sqrt(s2);
g = ( mu - fN_hat )./s;
a  = s.*(g.*normcdf(g)+normpdf(g));

if nargout>=2
    dadm  = 0.5*erfc( (fN_hat-mu)./(sqrt(2)*s) );
end

if nargout>=3
    dads2 = 0.5*normpdf(fN_hat,mu,s);
end


end
