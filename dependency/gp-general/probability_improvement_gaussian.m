function b = probability_improvement_gaussian( m, s2, cur_best )

s  = sqrt(s2);
ga = ( m - cur_best )./s;
b = normcdf(ga);

end
