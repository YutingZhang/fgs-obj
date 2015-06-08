
fprintf(1, 'Loading data list: \n'); 

fprintf(1, ' - Train : '); tic
TEST_DATA_LIST = GetDataList( PARAM.Train_DataSet );
[gtBoxes, gtImIds, CategList, ~, gtDifficulties, DifficultyList] = bboxes_from_list( TEST_DATA_LIST );
imageN = length(TEST_DATA_LIST);
toc

% get boxes
boxes = GetProposedBoxes( PARAM.Train_DataSet, SPECIFIC_DIRS.RegionProposal4GPTrain );
boxes = vec(boxes).';
toc

if length(boxes) ~= length( TEST_DATA_LIST );
    error( 'num of boxes is not consistent with the list' );
end

% load model
fprintf(1, 'Loading model : '); tic
classifier_model = GetClassifier( PARAM.CategOfInterest, SPECIFIC_DIRS.Train );
toc

%% test

TMP_CACHE_DIR = fullfile( SPECIFIC_DIRS.Test4GPTrain, 'cache' );
mkdir_p(TMP_CACHE_DIR);

for k = 1:imageN
    
    [~,fn,ext] = fileparts( TEST_DATA_LIST(k).im );
    fprintf(1,'%d / %d ( %4d ): %s : ', k, imageN, k, [fn ext] ); tic
    RESULT_FILE = fullfile( TMP_CACHE_DIR, [fn '.mat'] );
    if exist( RESULT_FILE, 'file' )
        fprintf( 'Existed\n' );
        continue;
    end
        
    F = GetProposedFeature( PARAM.Train_DataSet, fn, SPECIFIC_DIRS.Features4GPTrain );
    F = cell2mat( F );
    scores = ApplyClassifier( F, classifier_model );
    
    save( RESULT_FILE, 'scores' );
    toc

end

fprintf( 'Consolidate results : ' ); tic
S = cell( length(PARAM.CategOfInterest), imageN );
for k = 1:imageN
    [~,fn,ext] = fileparts( TEST_DATA_LIST(k).im );
    RESULT_FILE = fullfile( TMP_CACHE_DIR, [fn '.mat'] );
    R = load( RESULT_FILE );
    S(:,k) = mat2cell( R.scores, ones(1,length(PARAM.CategOfInterest)), size(R.scores,2) );
end

for c = 1:length(PARAM.CategOfInterest)
    scores = S(c,:);
    save( fullfile(SPECIFIC_DIRS.Test4GPTrain, [PARAM.CategOfInterest{c} '.mat']), 'scores' );
end
toc

fprintf(1,'Done\n');

