% Pipline stage: RegionProposal4GPTrain
%% Start your code here
% Note: R-CNN use the old version of Selective Search for VOC2007
%  To make our results comparable with theirs, we also use the same version
%  In this case, the proposed regions will be downloaded from the website.

[~, is_redirected] = sysRedirectList( SPECIFIC_DIRS.RegionProposal4GPTrain );
if is_redirected
    fprintf( 'Existed\n' );
    return;
end

AdditionalRegion_Dir = fullfile( SPECIFIC_DIRS.RegionProposal4GPTrain, 'additional_regions' );
mkdir_p( AdditionalRegion_Dir );

for subset_idx = 1 % 1:length(PARAM.DataSubSet_Names)

    ds = PARAM.DataSubSet_Names{subset_idx};
    
    fprintf( 1, '=============== %s\n', ds );
    
    DIR4SUBSET = fullfile( AdditionalRegion_Dir, ds );
    mkdir_p( DIR4SUBSET );
    
    BOXES_FILE_PATH = fullfile( DIR4SUBSET, 'boxes.mat' );
    if exist( BOXES_FILE_PATH, 'file' )
        fprintf( 1, ' Existed\n');
        continue;
    end
    
    fprintf( 1, 'Get data list: ' ); tic
    DATA_LIST = GetDataList(ds);
    imageN = length(DATA_LIST);
    [gtBoxes,  gtImIds  ] = bboxes_from_list( DATA_LIST );
    toc

    fprintf( 1, 'Propose bboxes: ' ); tic

    all_boxes = cell(imageN,numel(PARAM.GP_ADDITIONAL_REGION));
    for r = 1:numel(PARAM.GP_ADDITIONAL_REGION)
        [ augBoxes, augImIds ] = bboxes_augmentation( gtBoxes, gtImIds, ...
            PARAM.GP_ADDITIONAL_REGION{r}.GTNeighbor_Num, 0, ...
            PARAM.GP_ADDITIONAL_REGION{r}.GTNeighbor_MaxCenterShift, ...
            PARAM.GP_ADDITIONAL_REGION{r}.GTNeighbor_MaxScaleShift );

        [~,boxes] = reorganize_gt_wrt_image( imageN, [], augImIds, augBoxes );
        all_boxes(:,r) = vec(boxes);
    end
    
    boxes = cell(imageN,1);
    for k = 1:imageN
        boxes{k} = unique( cat(1,all_boxes{k,:}), 'rows' );
    end
    
    toc

    
    fprintf( 1, ' - Compute statistics: ' );
    tic
    [gtBoxes, gtImIds, CategList] = bboxes_from_list( DATA_LIST );
    [boxAbo, boxMabo, boScores, avgNumBoxes] = BoxAverageBestOverlap(gtBoxes, gtImIds, boxes);
    toc
    
    fprintf( 1, ' - Save to file: ' );

    tic
    save( BOXES_FILE_PATH, ...
        'boxes', 'boxAbo', 'boxMabo', 'boScores', 'avgNumBoxes', ...
        'gtBoxes', 'gtImIds', 'CategList', '-v7.3' );
    toc

    fprintf(1,'Mean Average Best Overlap for the box-based locations: %.3f\n', boxMabo);

    
end

fid = fopen( fullfile(SPECIFIC_DIRS.RegionProposal4GPTrain,'redirect.list'), 'w' );
fprintf(fid, '%s\n', relativepath( SPECIFIC_DIRS.RegionProposal, SPECIFIC_DIRS.RegionProposal4GPTrain ) );
fprintf(fid, '%s\n', relativepath( AdditionalRegion_Dir, SPECIFIC_DIRS.RegionProposal4GPTrain ) );
fclose(fid);

