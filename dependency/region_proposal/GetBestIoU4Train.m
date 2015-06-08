function [bestIoU, bestGtIdx] = GetBestIoU4Train ( CategOfInterest, BestIoU4Train_SpecificDir )

if ( ~exist( 'BestIoU4Train_SpecificDir', 'var' ) || isempty(BestIoU4Train_SpecificDir) ) ...
        && evalin( 'caller', 'exist(''SPECIFIC_DIRS'',''var'')' )
    BestIoU4Train_SpecificDir = evalin( 'caller', 'SPECIFIC_DIRS.BestIoU4Train' );
end

if ischar( CategOfInterest )
    CategOfInterest = { CategOfInterest };
end

cN = length(CategOfInterest);

bestIoU   = cell(1,cN);
bestGtIdx = cell(1,cN);

for c = 1:cN
    BEST_IoU_PATH = fullfile( BestIoU4Train_SpecificDir, ...
        sprintf('%s.mat',CategOfInterest{c}) );
    MAT_CONTENT = load( BEST_IoU_PATH );
    bestIoU{c}   = vec(MAT_CONTENT.bestIoU);
    bestGtIdx{c} = vec(MAT_CONTENT.bestGtIdx);
    clear MAT_CONTENT
end
bestIoU = [ bestIoU{:} ].'; bestGtIdx = [ bestGtIdx{:} ].';

bestIoU   = cat1dim( bestIoU.', 2 );   bestIoU   = cellfun( @(a) {a.'}, bestIoU );
bestGtIdx = cat1dim( bestGtIdx.', 2 ); bestGtIdx = cellfun( @(a) {a.'}, bestGtIdx );

end
