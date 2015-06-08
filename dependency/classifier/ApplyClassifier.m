function scores = ApplyClassifier( feat, classifier_model )

if isfield( classifier_model, 'type' )
    classifier_type = classifier_model.type;
elseif evalin( 'caller', 'exist(''PARAM'',''var'')' )
    classifier_type = evalin( 'caller', 'PARAM.Classifier_Type' );
else
    error( 'I cannot classifier type info' );
end

switch classifier_type
    case {'svm-linear','svm-localization','svm-struct'}
        % feat: d*n
        %    w: c*d
        %    b: c*1
        w = double(classifier_model.w);
        if isfield( classifier_model, 'scale' )
            w = bsxfun( @times, w, classifier_model.scale );
        end
        scores = bsxfun( @plus, w * double(feat), double( classifier_model.bias ) );
    otherwise
        error( ['Unknown classifier type : ' classifier_type] );
end

scores( isnan(scores) ) = -inf;

end
