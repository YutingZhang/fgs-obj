function y = ApplyRegressor( feat, regress_model )

if isfield( regress_model, 'type' )
    regressor_type = regress_model.type;
% elseif evalin( 'caller', 'exist(''PARAM'',''var'')' )
%     classifier_type = evalin( 'caller', 'PARAM.Classifier_Type' );
else
    error( 'I cannot classifier type info' );
end

switch regressor_type
    case {'linear-ridge'}
        % feat: d*n
        %    w: d*o*c
        %    b: 1*o*c
        w = double(regress_model.w);
        if isfield( regress_model, 'scale' )
            w = w * regress_model.scale;
        end
        S = size( w );
        y = bsxfun( @plus, ...
            reshape( w, S(1), S(2)*S(3) ).' * double(feat), ...
            reshape( double(regress_model.bias), 1, S(2)*S(3) ).' );
        y = reshape( y, S(2),S(3), size(feat,2) );
        if size(y,3)>1
            y = shiftdim(y,2); % y = permute( y, [3 1 2] );
        else
            y = shiftdim(y,-1);
        end
    otherwise
        error( ['Unknown classifier type : ' regressor_type] );
end

end
