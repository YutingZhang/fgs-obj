function scores = GetTestScores( CategOfInterest, Test_SpecificDir )

if ( ~exist( 'Test_SpecificDir', 'var' ) || isempty(Test_SpecificDir) ) ...
        && evalin( 'caller', 'exist(''SPECIFIC_DIRS'',''var'')' )
    Test_SpecificDir = evalin( 'caller', 'SPECIFIC_DIRS.Test' );
end

[Redirected_Dirs, is_redirected] = sysRedirectList( Test_SpecificDir );

if is_redirected
    
    sub_num = length(Redirected_Dirs);
    
    S = cell(sub_num,1);
    for sub_idx = 1:sub_num
        S{sub_idx} = GetTestScores( CategOfInterest, Redirected_Dirs{sub_idx} );
    end
    S = cat(1,S{:});
    scores = cat1dim( S, 1 );
        
else
    
    if ischar( CategOfInterest )
        CategOfInterest = { CategOfInterest };
    end
    
    cN = length(CategOfInterest);

    scores = cell(cN,1);
    for c=1:cN
        MAT_CONTENT = load( fullfile( Test_SpecificDir, [CategOfInterest{c} '.mat'] ) );
        scores{c} = MAT_CONTENT.scores;
    end

    scores = cat( 1, scores{:} );
    scores = mat2cell( scores, cN, ones(size(scores,2),1) );
    scores = cellfun( @(a) {double(cell2mat(a).')}, scores );
    for k=1:length(scores)
        scores{k}(isnan(scores{k})) = -inf;
        scores{k}(isnan(scores{k})>1e5) = 0;
    end

end
