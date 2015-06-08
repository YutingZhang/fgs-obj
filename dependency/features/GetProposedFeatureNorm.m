function feat_norm = GetProposedFeatureNorm( list_type, fn, Features4Proposed_SpecificDir, idxFeatType )

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
    try
        feat_norm = GetProposedFeatureNorm( list_type, fn, Parent_Dir, ...
            sub2ind( [Child_Shape 1], iip{:} ) );
        return;
    catch
    end
end

[Redirected_Dirs, is_redirected, Redirection_Shape] = ...
    sysRedirectList( Features4Proposed_SpecificDir, 2 );

if is_redirected
    
    sub_num = length(Redirected_Dirs);
    
    Fnorm = cell(1,sub_num);
    for sub_idx = 1:sub_num
        Fnorm{sub_idx} = GetProposedFeatureNorm( list_type, fn,  Redirected_Dirs{sub_idx} );
    end
    Fnorm = reshape(Fnorm,Redirection_Shape).';
    Fnorm = cat1dim( Fnorm,1 );
    feat_norm = cat( 2, Fnorm{:} );
    
    if nargout>1
        error( 'additional_info is currently not available' );
    end
    
else

    [~,fn,~] = fileparts(fn);

    featPath = fullfile(Features4Proposed_SpecificDir, list_type, [fn '.mat']);
    varFT = whos('-file',featPath,'numFeatType');
    if isempty(varFT)
        MAT_CONTENT = load( featPath, 'Fnorm' );
        if isempty( idxFeatType )
            feat_norm = MAT_CONTENT.Fnorm;
        else
            feat_norm = MAT_CONTENT.Fnorm(idxFeatType,:);
        end
    else
        if isempty( idxFeatType )
            MAT_CONTENT = load( featPath, 'numFeatType' );
            numFeatType = MAT_CONTENT.numFeatType;
            idxFeatType = 1:numFeatType;
        end
        idxFeatType = reshape(idxFeatType,1,numel(idxFeatType));
            
        VAR_TO_LOAD = arrayfun( ...
            @(r) {sprintf('Fnorm_%d',r)}, idxFeatType );
        MAT_CONTENT = load( featPath, VAR_TO_LOAD{:} );
        
        feat_norm = cell(length(idxFeatType),1);
        for rk = 1:length(idxFeatType)
            feat_norm{rk} = eval( sprintf( 'MAT_CONTENT.Fnorm_%d', idxFeatType(rk) ) );
        end
        feat_norm = cell2mat(feat_norm);
    end
    
end

end
