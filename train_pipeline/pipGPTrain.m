% Pipline stage: GPTrain

%% Parameters
% define parameters that do not affect output here
% e.g. OTHER_DEFAULT_PARAM.A = 0

% load data list
fprintf(1, 'Loading data list: '); tic
TRAIN_DATA_LIST = GetDataList( PARAM.Train_DataSet );
[gtBoxes, gtImIds, CategList ] = bboxes_from_list( TRAIN_DATA_LIST );
imageN = length(TRAIN_DATA_LIST);
toc

% Load bboxes
fprintf( 1, 'Load bboxes : ' );
tic
boxes = GetProposedBoxes( PARAM.Train_DataSet, SPECIFIC_DIRS.RegionProposal4GPTrain );
boxes = vec(boxes).';
toc

% Train on different categories
for c = 1:length(PARAM.CategOfInterest)
    fprintf( 1, '=========== %s\n', PARAM.CategOfInterest{c} );
    
    GP_MODEL_PATH = fullfile( SPECIFIC_DIRS.GPTrain, [PARAM.CategOfInterest{c} '.mat'] );
    if exist( GP_MODEL_PATH, 'file' )
        fprintf( 1, 'Existed\n' );
        continue;
    end
    
    fprintf( 1, 'Get scores : ' ); tic
    scores = GetTestScores( PARAM.CategOfInterest(c), SPECIFIC_DIRS.Test4GPTrain );
    toc
    
    fprintf( 1, 'Setup training data : ' ); tic
    [~,gt_boxes] = reorganize_gt_wrt_image( ...
        imageN, [], gtImIds{CategIdOfInterest(c)}, gtBoxes{CategIdOfInterest(c)} );
    gt_boxes = gt_boxes.';
    box_nums = cellfun( @(a) size(a,1), gt_boxes );
    if sum(box_nums) > PARAM.GP_Train_MaxLocalityNum
        box_cumnums = cumsum(box_nums);
        chosenImageN = find( box_cumnums <= PARAM.GP_Train_MaxLocalityNum, 1, 'last' );
    else
        chosenImageN = imageN;
    end
    
    numAvailBoxes = sum( box_nums(1:chosenImageN) );
    
    BoxGroups   = cell(numAvailBoxes, 1);
    ScoreGroups = cell(numAvailBoxes, 1);
    
    AnchorInstances = nan(numAvailBoxes, 4); % for latent variable
    
    b = 0;
    for k = 1:chosenImageN
        
        cur_IoU = PairedIoU( boxes{k}, gt_boxes{k} );
        
        numGT_k = size(gt_boxes{k},1);
        
        for j = 1:numGT_k
            b = b + 1;

            localIdxB = (cur_IoU(:,j)>PARAM.GP_Local_IoU_Threshold);
        
            if sum(localIdxB)>PARAM.GP_Local_MaxSampleNum
                chosenIdx = find( localIdxB );
                localIdxB( chosenIdx( PARAM.GP_Local_MaxSampleNum+1 ):end ) = false;
                clear chosenIdx
            end
            BoxGroups{b}   = boxes{k}( localIdxB, : );
            ScoreGroups{b} = scores{k}( localIdxB );
            
            AnchorInstances(b,:) = gt_boxes{k}(j,:);

            [BoxGroups{b}, ia] = unique( BoxGroups{b}, 'rows', 'stable' );
            ScoreGroups{b} = ScoreGroups{b}(ia);
        end
    end
    
    trainX = cellfun( @(a) { bbox_ltrb2param( a, PARAM.GP_BBoxParamType ) }, BoxGroups );

    [~,~,gtSl,~] = bbox_ltrb2param( AnchorInstances, 'yxsal' );
    gtS = exp(gtSl);
    for b = 1:numAvailBoxes
        trainX{b}(:,[1 2]) = trainX{b}(:,[1 2]) / gtS(b);
    end

    trainY = ScoreGroups;
    
    toc
    
    fprintf( 1, 'Train : ' );
    
    GP_Param = struct();
    GP_Param.BBoxParamType = PARAM.GP_BBoxParamType;
    GP_Param.LatentScaling_Enabled = 1;
    
    GP_Param.MeanFunc = PARAM.GP_MeanFunc;
    GP_Param.CovFunc  = PARAM.GP_CovFunc;
    GP_Param.LikFunc  = PARAM.GP_LikFunc;
    
    meanfunc = eval( sprintf('{%s}', GP_Param.MeanFunc ) ); 
    covfunc  = eval( sprintf('{%s}', GP_Param.CovFunc  ) ); 
    likfunc  = eval( sprintf('{%s}', GP_Param.LikFunc  ) ); 
    
    x_dim = size( trainX{1}, 2 );
    hyp.mean = repmat( 0, 1, gp_hyp_dim( meanfunc, x_dim ) );
    hyp.cov  = repmat( 0, 1, gp_hyp_dim( covfunc,  x_dim ) );
    sn = 0.1; hyp.lik = repmat( log(sn), 1, gp_hyp_dim( likfunc, x_dim ) );
   
    try
        hyp2 = minFuncX( @(h) gp_cellInput(h, ...
            @infExact, meanfunc, covfunc, likfunc, trainX, trainY), hyp );
    catch
        warning('minFuncX failed, try the alternative solver : minimize_lbfgsb');
        try
            hyp2 = minimize_lbfgsb(hyp, @gp_cellInput, -PARAM.GP_Train_MaxIterationNum, ...
                @infExact, meanfunc, covfunc, likfunc, trainX, trainY );
        catch
            warning('minimize_lbfgsb failed, try the alternative solver : minimize');
            hyp2 = minimize(hyp, @gp_cellInput, -PARAM.GP_Train_MaxIterationNum, ...
                @infExact, meanfunc, covfunc, likfunc, trainX, trainY );
        end
    end

    GP_Param.hyp = hyp2;
    
    save( GP_MODEL_PATH, '-struct', 'GP_Param' );
    
    toc
    
    
end
