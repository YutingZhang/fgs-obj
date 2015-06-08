function [ll, dll_dCnoisy] = sgp_negloglik_givenC( GPmodel, C, f )

n = length(f);
fn = f - GPmodel.m0;
ll = 0.5 * ( log( (2*pi)^n * det(C) ) + (fn.'/C) * fn );

if nargout>=2
    dll_dCnoisy = -0.5 * ((C\fn)*(fn.'/C)-inv(C));
end

end
