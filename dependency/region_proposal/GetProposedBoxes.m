function [boxes, additional_info] = GetProposedBoxes( list_type, RegionProposal_SpecificDir )

if ( ~exist( 'RegionProposal_SpecificDir', 'var' ) || isempty(RegionProposal_SpecificDir) ) ...
        && evalin( 'caller', 'exist(''SPECIFIC_DIRS'',''var'')' )
    RegionProposal_SpecificDir = evalin( 'caller', 'SPECIFIC_DIRS.RegionProposal' );
end

[Redirected_Dirs, is_redirected] = sysRedirectList( RegionProposal_SpecificDir );

if is_redirected
    
    sub_num = length(Redirected_Dirs);
    
    S = cell(sub_num,1);
    for sub_idx = 1:sub_num
        S{sub_idx} = GetProposedBoxes( list_type, Redirected_Dirs{sub_idx} );
    end
    S = cellfun( @(a) {vec(a).'}, S );
    S = cat(1,S{:});
    boxes = cat1dim( S, 1 );
    
    if nargout>1
        error( 'additional_info is currently not available' );
    end
    
else


    BOXES_NAME = fullfile( RegionProposal_SpecificDir, list_type, 'boxes.mat' );
    MAT_CONTENT = load(BOXES_NAME);
    boxes = MAT_CONTENT.boxes;

    additional_info = rmfield( MAT_CONTENT, 'boxes' );


end
