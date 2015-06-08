function classifier_model = GetClassifier( CategOfInterest, Train_SpecificDir )

if ( ~exist( 'Train_SpecificDir', 'var' ) || isempty(Train_SpecificDir) ) ...
        && evalin( 'caller', 'exist(''SPECIFIC_DIRS'',''var'')' )
    Train_SpecificDir = evalin( 'caller', 'SPECIFIC_DIRS.Train' );
end

cN = length(CategOfInterest);
CM = cell(cN,1);
for c = 1:cN
    CM{c} = load( fullfile(Train_SpecificDir, [CategOfInterest{c}, '.mat']) );
end
CM = cell2mat(CM);

if isfield( CM, 'type' )
    classifier_type = CM(1).type;
elseif evalin( 'caller', 'exist(''PARAM'',''var'')' )
    classifier_type = evalin( 'caller', 'PARAM.Classifier_Type' );
else
    error( 'I cannot get classifier type info' );
end

% merge multiple classifiers
classifier_model.type = classifier_type;
switch classifier_type
    case 'identical'
        classifier_model.dim_idx = cat( 1, CM.dim_idx );
    case {'svm-linear','svm-localization','svm-struct'}
        if isfield( classifier_model, 'scale' )
            if ~isequal({CM.scale}, repmat({CM(1).scale},1,length(CM)))
                error( 'Inconsistent scaling factor for features' );
            end
%             classifier_model.scale = CM(1).scale;
%         else
%             classifier_model.scale = 1;
        end
        classifier_model.w     = bsxfun( @times, cat( 1, CM.w ), CM(1).scale);  % normalize scale to 1
        classifier_model.bias  = cat( 1, CM.bias );
    otherwise
        error( ['Unknown classifier type : ' classifier_type] );
end

end
