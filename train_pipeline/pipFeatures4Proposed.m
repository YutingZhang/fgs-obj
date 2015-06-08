% Pipline stage: Features4Proposed

PARAM.Collection_BatchNum = 5;
PARAM.Feature_BatchSize = 32;

Caffe_Layer = {'pool5','fc7'};
scriptSetupPatchFeatures;

collection_size = batchsize_for_feature_extractor * PARAM.Collection_BatchNum;

for subset_idx = 1:length(PARAM.DataSubSet_Names)

    ds = PARAM.DataSubSet_Names{subset_idx};

    fprintf( 1, '========================= %s \n', ds );
    
    DATA_LIST = GetDataList(ds);
    imageN = length( DATA_LIST );
    
    DIR4SUBSET = fullfile( SPECIFIC_DIRS.Features4Proposed, ds ); 
    mkdir_p( DIR4SUBSET );
    
    if ~isempty(SPECIFIC_DIRS.Features4Proposed_bboxreg)
        DIR4SUBSET_bboxreg = fullfile( SPECIFIC_DIRS.Features4Proposed_bboxreg, ds ); 
        mkdir_p( DIR4SUBSET_bboxreg );
    end
    
    fprintf( 1, 'Load boxes : ' ); tic
    boxes = GetProposedBoxes( ds );
    toc

    for k=1:imageN

        [~,fn,ext] = fileparts( DATA_LIST(k).im );
        fprintf( 1, '%d / %d : %s : %d patches : ', ...
            k, imageN, [fn,ext], size(boxes{k},1) );
        FEATURE_FILE = fullfile( DIR4SUBSET, [fn '.mat'] );
        if ~isempty(SPECIFIC_DIRS.Features4Proposed_bboxreg)
            FEATURE_FILE_bboxreg = fullfile( DIR4SUBSET_bboxreg, [fn '.mat'] );
        end
        % is file exist?
        if exist(FEATURE_FILE, 'file') && ...
                ( isempty(SPECIFIC_DIRS.Features4Proposed_bboxreg) || ...
                exist(FEATURE_FILE_bboxreg, 'file') )
            fprintf(1,'Existed\n');
        else

            tic
            I = imread( DATA_LIST(k).im );

            F = features_from_bboxes( I, boxes{k}, canonical_patchsize, ...
                PARAM.Patch_Padding, feat_func, collection_size );
            Fnorm = cellfun( @(a) {sqrt(sum(a.*a,1))}, F );
            % Fnorm = cell2mat(Fnorm);
            
            if ~isempty(SPECIFIC_DIRS.Features4Proposed_bboxreg)
                C = struct( 'F', {F(1)}, 'Fnorm', Fnorm(1) );
                save( FEATURE_FILE_bboxreg, '-struct', 'C', '-v7.3' );
            end
            C = struct( 'F', {F(2)}, 'Fnorm', Fnorm(2) );
            save( FEATURE_FILE, '-struct', 'C', '-v7.3' );
            toc

        end
    end
    
    
end
