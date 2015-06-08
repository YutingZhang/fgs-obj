% Pipline stage: RegionProposal
%% Start your code here
% Note: R-CNN use the old version of Selective Search for VOC2007
%  To make our results comparable with theirs, we also use the same version
%  In this case, the proposed regions will be downloaded from the website.
PARAM.USE_OLD_TOOLBOX = 1;
PARAM.EachBatch_Size = 144;

SelectiveSearchInit();
rp_func = @(im) SelectiveSearchOnOneImage( im, 'ijcv_fast' );

for subset_idx = 1:length(PARAM.DataSubSet_Names)

    ds = PARAM.DataSubSet_Names{subset_idx};
    
    fprintf( 1, '=============== %s\n', ds );
    
    DIR4SUBSET = fullfile( SPECIFIC_DIRS.RegionProposal, ds );
    mkdir_p( DIR4SUBSET );
    
    BOXES_FILE_PATH = fullfile( DIR4SUBSET, 'boxes.mat' );
    if exist( BOXES_FILE_PATH, 'file' )
        fprintf( 1, ' Existed\n');
        continue;
    end
    
    if PARAM.USE_OLD_TOOLBOX
        % Download data
        
        fprintf( 1, 'Download data: \n' );
        TRAINVAL_PATH = fullfile( SPECIFIC_DIRS.RegionProposal, 'trainval_old.mat' );
        TEST_PATH     = fullfile( SPECIFIC_DIRS.RegionProposal, 'test_old.mat' );
        TRAINVAL_URL = 'http://www.huppelen.nl/SelectiveSearch/SelectiveSearchVOC2007trainval.mat';
        TEST_URL     = 'http://www.huppelen.nl/SelectiveSearch/SelectiveSearchVOC2007test.mat';
        
        fprintf( 1, ' - Get trainval data: ' );
        if exist( TRAINVAL_PATH, 'file' )
            fprintf( 1, 'Existed\n' );
        else
            url2file( TRAINVAL_URL, [TRAINVAL_PATH '.tmp'], '-c' );
            movefile( [TRAINVAL_PATH '.tmp'], TRAINVAL_PATH );
        end
        fprintf( 1, ' - Get test data: ' );
        if exist( TEST_PATH, 'file' )
            fprintf( 1, 'Existed\n' );
        else
            url2file( TEST_URL, [TEST_PATH '.tmp'], '-c' );
            movefile( [TEST_PATH '.tmp'], TEST_PATH );
        end
        
        if ismember( ds, {'train','val','trainval'} )
            OLD_DATA = load( TRAINVAL_PATH );
        elseif ismember( ds, {'test'} )
            OLD_DATA = load( TEST_PATH );
        else
            error( 'Unsupported list_type' );
        end
        
        fprintf( 1, 'Retrieve data: ' );
        tic
        DATA_LIST = GetDataList( ds );
        imNames = {DATA_LIST.im};
        for k = 1:length(imNames)
            [~,imNames{k},~] = fileparts(imNames{k});
        end
        [list_availability, list_mapping] = ismember( ...
            imNames, OLD_DATA.images );
        if ~all(list_availability)
            error( 'Not all the entries are present' );
        end
        boxes = OLD_DATA.boxes(list_mapping);
        toc
        
    else
        DATA_LIST = GetDataList(ds);

        fprintf( 1, 'Propose bboxes: \n' );
        
        imageN = length(DATA_LIST);
        
        boxes = cell(imageN,1);
        
        b = 0;
        k = 1;
        while k <= imageN
            b=b+1;
            fprintf(1, ' - Batch %d: \n', b );
            fprintf(1, '   * Loading images : ' );

            tic
            
            num_in_batch = 0;
            IM    = cell(PARAM.EachBatch_Size,1);
            imIdx = zeros(PARAM.EachBatch_Size,1);
            
            k_begin = k;
            while k <= imageN
                num_in_batch = num_in_batch+1;
                imIdx(num_in_batch) = k;
                IM{num_in_batch} = imread( DATA_LIST(k).im );
                k = k + 1;
                if num_in_batch>=PARAM.EachBatch_Size
                    break;
                end
            end
            fprintf( 1, '%d out of %d - %d / %d : ', num_in_batch, k_begin, k-1 , imageN );
            toc
            
            fprintf(1, '   * Proposing regions : ' ); tic
            IM   ( num_in_batch+1:end ) = [];
            imIdx( num_in_batch+1:end ) = [];
            B = cell( num_in_batch, 1 );
            parfor i=1:length(IM)
                B{i} = rp_func( IM{i} );
            end
            boxes(imIdx) = B;
            toc
            
        end
    end

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


