% Features4Groundtruth

PARAM.Collection_BatchNum = 5;
PARAM.Feature_BatchSize = 32;

Caffe_Layer = 'fc7';

scriptSetupPatchFeatures;   % 

batchsize = batchsize_for_feature_extractor;
batchnum_per_collection = PARAM.Collection_BatchNum;

for subset_idx = 1 % 1:length(PARAM.DataSubSet_Names)

    ds = PARAM.DataSubSet_Names{subset_idx};    

    fprintf( 1, '========================= %s \n', ds );

    existed_categ_idxb = cellfun( @(a) ...
        exist(fullfile(SPECIFIC_DIRS.Features4Groundtruth, ds,[a '.mat']),'file') , PARAM.CategOfInterest );
    if all(existed_categ_idxb), fprintf('Existed\n'), continue; end
    
    
    DATA_LIST = GetDataList(ds);
    imageN = length( DATA_LIST );
    
    DIR4SUBSET = fullfile( SPECIFIC_DIRS.Features4Groundtruth, ds );
    mkdir_p( DIR4SUBSET );
    
    [gtBoxes, gtImIds, CategList] = bboxes_from_list( DATA_LIST );

    % reorganizing bboxes
    fprintf( 1, 'Recoganizing bboxes : ' );

    tic

    [caIds, boxes] = reorganize_gt_wrt_image(imageN, CategIdOfInterest, gtImIds, gtBoxes);

    toc

    % image batch partition

    numBoxes = cellfun( @length, caIds );
    [imgCollectionSplitter, imgBatchSampleNum ] = ...
        vec_partition_to_fixed(numBoxes, batchsize * batchnum_per_collection);
    imgCollectionNum = length(imgBatchSampleNum);

    % extract feature in batch

    fprintf(1, 'Extract features : \n');

    gtF = cell( 1, imgCollectionNum, length(CategList) );
    for i = 1:imgCollectionNum
        fprintf( 1, 'Collection %d / %d : \n', i, imgCollectionNum );

        % prepare enough patches
        fprintf( 1, ' - Loading images : ' );
        tic

        st = imgCollectionSplitter(i,1); en = imgCollectionSplitter(i,2);
        P = repmat( {uint8([])},[1,1,1,imgBatchSampleNum(i)] );
        for k = st:en
            if ~isempty(boxes{k})
                I = imread( DATA_LIST(k).im );
                P{k-st+1} = extract_patches_from_image( single(I), boxes{k}, ...
                    canonical_patchsize, PARAM.Patch_Padding );
            end
        end
        P = cell2mat(P);
        cur_Categ = cell2mat( caIds(st:en) );

        toc

        cur_image_patch_num = size(P,4);

        % extract features
        fprintf( 1, ' - Extracting features : ' );
        tic
        F = feat_func( P );
        toc

        % put feature into different categories
        fprintf( 1, ' - Re-organizing features : ' );
        tic
        for c = CategIdOfInterest
            chosenIdx = (cur_Categ==c);
            gtF(:,i,c) = arrayfun( @(a) { a{1}(:,chosenIdx) }, F );
        end
        toc

    end

    fprintf(1, 'Re-organizing features : ');
    tic

    gtF = mat2cell( gtF, ones(size(gtF,1),1), size(gtF,2), ones(size(gtF,3),1) );
    gtF = reshape( gtF, [size(gtF,1),size(gtF,3)] );
    gtF = arrayfun( @(a) {cell2mat(a{1})}, gtF );

    toc

    fprintf(1, 'Save features : \n');
    for c = CategIdOfInterest
        fprintf( 1, ' - %s : ', CategList{c} );
        tic
        F = gtF(:,c);
        ImIds = gtImIds{c};
        Boxes  = gtBoxes{c};

        feat_fn_c = fullfile( DIR4SUBSET, [CategList{c} '.mat'] );
        save(feat_fn_c,'F','ImIds','Boxes');
        toc
    end

end
