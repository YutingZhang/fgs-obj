function regressor_model = GetRegressor( CategOfInterest, BBoxRegTrain_SpecificDir )

if ( ~exist( 'BBoxRegTrain_SpecificDir', 'var' ) || isempty(BBoxRegTrain_SpecificDir) ) ...
        && evalin( 'caller', 'exist(''SPECIFIC_DIRS'',''var'')' )
    BBoxRegTrain_SpecificDir = evalin( 'caller', 'SPECIFIC_DIRS.BBoxRegTrain' );
end

cN = length(CategOfInterest);
CM = cell(cN,1);
for c = 1:cN
    CM{c} = load( fullfile(BBoxRegTrain_SpecificDir, [CategOfInterest{c}, '.mat']) );
end
CM = cell2mat(CM);

if isfield( CM, 'type' )
    regressor_type = CM(1).type;
elseif evalin( 'caller', 'exist(''PARAM'',''var'')' )
    regressor_type = evalin( 'caller', 'PARAM.Classifier_Type' );
else
    error( 'I cannot get regressor type info' );
end

% merge multiple classifiers
regressor_model.type = regressor_type;
switch regressor_type
    case {'linear-ridge'}
        if isfield( regressor_model, 'scale' )
            regressor_model.scale = unique([CM.scale]);
        else
            regressor_model.scale = 1;
        end
        regressor_model.w     = cat( 3, CM.w );
        regressor_model.bias  = cat( 3, CM.bias );
        if numel(regressor_model.scale)>1
            error( 'Inconsistent scaling factor for features' );
        end
    otherwise
        error( ['Unknown classifier type : ' regressor_type] );
end

end
