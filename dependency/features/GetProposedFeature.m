function [feat, full_index] = GetProposedFeature( list_type, fn, ...
    Features4Proposed_SpecificDir, idxFeatType )

if ( ~exist( 'Features4Proposed_SpecificDir', 'var' ) || isempty(Features4Proposed_SpecificDir) ) ...
        && evalin( 'caller', 'exist(''SPECIFIC_DIRS'',''var'')' )
    Features4Proposed_SpecificDir = evalin( 'caller', 'SPECIFIC_DIRS.Features4Proposed' );
end

if ~exist('idxFeatType','var')
    idxFeatType = [];
end

[Parent_Dir, has_parent, Index_In_Parent, Child_Shape] = sysParentLink( Features4Proposed_SpecificDir );
if has_parent
    if ~isempty( idxFeatType ) && idxFeatType~=1
        error( 'Do not support idxFeatType for child-stage' );
    end
    iip = num2cell(Index_In_Parent);

    [feat, full_index] = GetProposedFeature( list_type, fn, Parent_Dir, ...
        sub2ind( [Child_Shape,1], iip{:} ) ); %[...,1] prevent error
    return;

end

[Redirected_Dirs, is_redirected, Redirection_Shape] = ...
    sysRedirectList( Features4Proposed_SpecificDir, 2 );

if is_redirected
    
    sub_num = length(Redirected_Dirs);
    
    F = cell(1,sub_num);
    for sub_idx = 1:sub_num
        [F{sub_idx} full_index] = GetProposedFeature( list_type, fn, Redirected_Dirs{sub_idx}, [] );
        % full_idx need to be updated here (maybe not very useful)
    end
    
    F = reshape(F,[Redirection_Shape 1]).'; % first combined region, then features
       
    F = cat1dim( F, 1 );
    F = cat( 2, F{:} );
    F = cat1dim( F, 2 );
    feat = F;
    
    if nargout>1
        error( 'additional_info is currently not available' );
    end
    
else

    [~,fn,~] = fileparts(fn);

    DIR4SUBSET = fullfile(Features4Proposed_SpecificDir, list_type );
    
    featPath = fullfile( DIR4SUBSET, [fn '.mat']);
    
    feat = get_feat_general_from_mat( featPath, 'F', idxFeatType );
    full_index = 1:size(feat{1},2);

    
    for i = 1:numel(feat)
        feat{i}(isnan(feat{i}(:)) | abs(feat{i}(:))>1e3 ) = 0;
    end
    
end

end


function F = get_feat_general_from_mat( featPath, varPrefix, idxFeatType )

varFT  = whos('-file',featPath,'numFeatType');

if isempty(varFT)
    MAT_CONTENT = load( featPath, varPrefix );
    if isempty( idxFeatType )
        F = eval( sprintf( 'MAT_CONTENT.%s', varPrefix ) );
    else
        F = eval( sprintf( 'MAT_CONTENT.%s(idxFeatType,:)', varPrefix ) );
    end
else
    if isempty( idxFeatType )
        MAT_CONTENT = load( featPath, 'numFeatType' );
        numFeatType = MAT_CONTENT.numFeatType;
        idxFeatType = 1:numFeatType;
    end
    idxFeatType = reshape(idxFeatType,1,numel(idxFeatType));

    VAR_TO_LOAD = arrayfun( ...
        @(r) {sprintf('%s_%d',varPrefix, r)}, idxFeatType );
    MAT_CONTENT = load( featPath, VAR_TO_LOAD{:} );

    F = cell(length(idxFeatType),1);
    for rk = 1:length(idxFeatType)
        F(rk) = eval( sprintf( 'MAT_CONTENT.%s_%d', ...
            varPrefix, idxFeatType(rk) ) );
    end
end

end
