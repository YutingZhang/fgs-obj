function b = expected_improvement_gaussian( m, s2, cur_best )

s  = sqrt(s2);
ga = ( m - cur_best )./s;
b  = s.*(ga.*normcdf(ga)+normpdf(ga));

end
