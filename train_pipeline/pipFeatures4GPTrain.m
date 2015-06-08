

Features4AdditionalRegion_Dir = fullfile( SPECIFIC_DIRS.Features4GPTrain, 'additional_regions' );
mkdir_p( Features4AdditionalRegion_Dir );

SPECIFIC_DIRS0   = SPECIFIC_DIRS;
SPECIFIC_DIRSgp  = SPECIFIC_DIRS;
SPECIFIC_DIRSgp.Features4Proposed = Features4AdditionalRegion_Dir;
SPECIFIC_DIRSgp.Features4Proposed_bboxreg = [];
SPECIFIC_DIRSgp.RegionProposal    = fullfile( SPECIFIC_DIRSgp.RegionProposal4GPTrain, ...
    'additional_regions' );

SPECIFIC_DIRS = SPECIFIC_DIRSgp;
pipFeatures4Proposed

SPECIFIC_DIRS = SPECIFIC_DIRS0;

fid = fopen( fullfile(SPECIFIC_DIRS.Features4GPTrain,'redirect.list'), 'w' );
fprintf(fid, '%s\n', relativepath( SPECIFIC_DIRS.Features4Proposed, SPECIFIC_DIRS.Features4GPTrain ) );
fprintf(fid, '%s\n', relativepath( Features4AdditionalRegion_Dir, SPECIFIC_DIRS.Features4GPTrain ) );
fclose(fid);
