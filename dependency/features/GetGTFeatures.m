function GTFeat = GetGTFeatures( list_type, CategName, Features4Groundtruth_SpecificDir )

if ( ~exist( 'Features4Groundtruth_SpecificDir', 'var' ) || isempty(Features4Groundtruth_SpecificDir) ) ...
        && evalin( 'caller', 'exist(''SPECIFIC_DIRS'',''var'')' )
    Features4Groundtruth_SpecificDir = evalin( 'caller', 'SPECIFIC_DIRS.Features4Groundtruth' );
end

[Redirected_Dirs, is_redirected] = sysRedirectList( Features4Groundtruth_SpecificDir );

if is_redirected
    sub_num = length(Redirected_Dirs);
    
    G = cell(1,sub_num);
    for sub_idx = 1:sub_num
        G{sub_idx} = GetGTFeatures( list_type, CategName, Redirected_Dirs{sub_idx} );
    end
    G = cell2mat(G);
    
    % check boxes
    Boxes = cat(3,G.Boxes);
    if ~isequal( max(Boxes,[],3), min(Boxes,[],3) )
        error( 'GetGTFeatures : "Boxes" should be consistent along differen subpipeline' ); 
    end
    % check ImIds
    ImIds = cat(3,G.ImIds);
    if ~isequal( max(ImIds,[],3), min(ImIds,[],3) )
        error( 'GetGTFeatures : "ImIds" should be consistent along differen subpipeline' ); 
    end
    % merge features
    F = cat(1, G.F );
    
    GTFeat = struct( 'Boxes', {G(1).Boxes}, 'ImIds', {G(1).ImIds}, 'F', {F} );
else
    GTFeat = load( fullfile( Features4Groundtruth_SpecificDir, list_type, [CategName '.mat'] ) );
end


end
