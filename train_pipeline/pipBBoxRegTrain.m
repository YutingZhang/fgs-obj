% Pipline stage: BBoxRegTrain

%% Start your code here

%% initialization

% Load Data List
fprintf(1, 'Loading data list: '); tic
TRAIN_DATA_LIST = GetDataList( PARAM.Train_DataSet );
[gtBoxes, gtImIds, CategList, ~, gtDifficulties, DifficultyList] = bboxes_from_list( TRAIN_DATA_LIST );
imageN = length(TRAIN_DATA_LIST);
toc

% existed?

existed_categ_idxb = cellfun( @(a) ...
    exist(fullfile(SPECIFIC_DIRS.BBoxRegTrain,[a '.mat']),'file') , PARAM.CategOfInterest );
if all(existed_categ_idxb), fprintf('Existed\n'), return; end

% Load Proposed Boxes
fprintf(1, 'Loading Proposed Boxes: '); tic
boxes = GetProposedBoxes( PARAM.Train_DataSet );
boxes = vec(boxes).';
toc

% Load Best IoU
fprintf(1, 'Load Best IoU: '); tic
[bestIoU, bestGtIdx] = GetBestIoU4Train( PARAM.CategOfInterest );
toc

% Load the mean norm of the proposed samples
fprintf(1, 'Load Mean Feature Norm: '); tic
mean_feat_norm = load( fullfile( SPECIFIC_DIRS.FeatureNorm4Train_bboxreg, 'mean_feat_norm.mat' ) );
mean_feat_norm = mean_feat_norm.mean_feat_norm;
toc
feat_scale_factor = PARAM.Feature_Norm / mean_feat_norm;
fprintf( 1, 'Feature Scaling Factor: %g\n', feat_scale_factor );

% re-organize gt boxes
fprintf(1, 'Re-organize GT boxes: '); tic
gt_boxes = cell(length(CategIdOfInterest),imageN);
for c = 1:length(CategIdOfInterest)
    [~, gt_boxes(c,:)]= reorganize_gt_wrt_image( ...
        imageN,CategIdOfInterest(c), gtImIds, gtBoxes );
end
toc

%% features & regression target

fprintf( '==== Load features and generate regression targets : \n' );

X = cell(1,imageN);
V = cell(1,imageN);
Y = cell(length(CategIdOfInterest),imageN);
for k = 1:imageN

    [~,fn,ext] = fileparts( TRAIN_DATA_LIST(k).im );
    fprintf( '%d / %d (%s) : ', k, imageN, [fn ext] );
    
    tic
    % load features
    F = GetProposedFeature( PARAM.Train_DataSet, TRAIN_DATA_LIST(k).im, ...
        SPECIFIC_DIRS.Features4Proposed_bboxreg );
    F = double( cell2mat( F ) ) * feat_scale_factor;
    
    % find the bboxes for training
    validBBoxIdxB = (bestIoU{k}>PARAM.BBoxRegression_IoUThreshold);
    
    % find the regression target
    for c = 1:length(CategIdOfInterest)
        gt_boxes_c = gt_boxes{c,k};
        gt_param_c = bbox_ltrb2param( gt_boxes_c, 'yxhw' );
        G = gt_param_c( bestGtIdx{k}(c,validBBoxIdxB(c,:)), : );
        P = bbox_ltrb2param( boxes{k}(validBBoxIdxB(c,:),:), 'yxhw' );
        
        Tyx = ( G(:,[1 2])-P(:,[1 2]) )./P(:,[3 4]);
        Thw = log( G(:,[3 4])./P(:,[3 4]) );
        Y{c,k} = [Tyx, Thw].';
    end
    
    % only keep the useful features
    semiValidBBoxIdxB = any( validBBoxIdxB, 1 );
    X{k} = F( :, semiValidBBoxIdxB );
    V{k} = validBBoxIdxB( :, semiValidBBoxIdxB );
    
    toc
    
end

X = [X{:}];
V = [V{:}];
Y = cat1dim( Y, 2 );

%% do regression

fprintf( '==== Do BBox regression : \n' );

for c = 1:length(CategIdOfInterest)
    fprintf( '%s : ', CategList{CategIdOfInterest(c)} ); tic
    Xo = [X(:,V(c,:)).', ones(sum(V(c,:)),1)];
    w = nan(size(Xo,2),size(Y{c},1));
    for d = 1:size(Y{c},1)
        w(:,d) = ridge_regress(Xo, Y{c}(d,:).', ...
            PARAM.BBoxRegression_Ridge_Lambda);
        fprintf( '.' );
    end
    fprintf( ' ' );
    regress_model.type    = 'linear-ridge';
    regress_model.w       = w(1:end-1,:) * feat_scale_factor;
    regress_model.bias    = w(end,:);
    regress_model.scaling = 1;
    
    MODEL_PATH = fullfile( SPECIFIC_DIRS.BBoxRegTrain, ...
        [ CategList{CategIdOfInterest(c)} '.mat'] ); 
    
    save( MODEL_PATH, '-struct', 'regress_model' ); 
    
    fprintf( ' Saved. ' );
    
    toc
end

fprintf( 'Done\n' );

