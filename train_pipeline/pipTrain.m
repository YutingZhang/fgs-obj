% Pipline stage: Train

%% Start your code here

%% Initialize 

% Load Data List
fprintf(1, 'Loading data list: '); tic
TRAIN_DATA_LIST = GetDataList( PARAM.Train_DataSet );
[gtBoxes, gtImIds, CategList, ~, gtDifficulties, DifficultyList] = bboxes_from_list( TRAIN_DATA_LIST );
imageN = length(TRAIN_DATA_LIST);
toc

existed_categ_idxb = cellfun( @(a) ...
    exist(fullfile(SPECIFIC_DIRS.Train,[a '.mat']),'file') , PARAM.CategOfInterest );
if all(existed_categ_idxb), fprintf('Existed\n'), return; end

% Load the mean norm of the proposed samples
fprintf(1, 'Load Mean Feature Norm: '); tic
mean_feat_norm = load( fullfile( SPECIFIC_DIRS.FeatureNorm4Train, 'mean_feat_norm.mat' ) );
mean_feat_norm = mean_feat_norm.mean_feat_norm;
toc

% Feature scaling and load func
feat_scale_factor = (PARAM.Feature_Norm / sqrt(length(mean_feat_norm))) ./ mean_feat_norm;
fprintf( 1, 'Feature Scaling Factor:') ; 
fprintf( 1, ' %g', feat_scale_factor ); fprintf(1,'\n');

feat_loading_func = @(k) GetProposedFeature( ...
    PARAM.Train_DataSet, TRAIN_DATA_LIST(k).im, SPECIFIC_DIRS.Features4Proposed );
feat_scaling_merger_func = @(cell_feat) ...
    cell2mat( cellfun( @(f,s) {f*s}, cell_feat, num2cell(feat_scale_factor) ) );
bag_func = @(k) feat_scaling_merger_func( feat_loading_func(k) );

feat_lengths   = cellfun( @(f) size(f,1), feat_loading_func(1) );
feat_scale_vec = cell2mat( arrayfun( @(s,l) {repmat(s,l,1)}, ...
    feat_scale_factor, feat_lengths ) );

% create feature cache for postive samples
Cache_Pos = cell(length(CategIdOfInterest),1);
ImIds_Pos = cell(length(CategIdOfInterest),1);

% Load Pos Features

fprintf(1, 'Loading Pos Features: '); tic

for c = 1:length(CategIdOfInterest)
    GTFeat = GetGTFeatures( PARAM.Train_DataSet, PARAM.CategOfInterest{c} );
    Cache_Pos{c} = feat_scaling_merger_func( GTFeat.F );
    ImIds_Pos{c} = GTFeat.ImIds;
    clear GTFeat
end

toc

% Load Best IoU
fprintf(1, 'Load Best IoU: '); tic
[bestIoU, bestGtIdx] = GetBestIoU4Train( PARAM.CategOfInterest );
toc


%% Train and Hard mining

Epoch_Callback = @(cl, ep) SaveClassifier( ...
    modify_struct( cl, 'scale',feat_scale_vec.' ), ...
    PARAM.Classifier_Type, SPECIFIC_DIRS.Train, ...
    PARAM.CategOfInterest, sprintf('epoch_%d',ep) );

if strcmp( PARAM.Classifier_Type, 'svm-linear' )
    neg_bag_filter_func = @(c,k) find( ...
        bestIoU{k}(c,:)<=PARAM.Train_Neg_IoU_Threshold & ... 
        bestIoU{k}(c,:)<1);

    classifier_model_all = HardTrainSVM( Cache_Pos, 2000, PARAM.SVM_Epoch, imageN, ...
        bag_func, neg_bag_filter_func, ...
        1, 'liblinear', ...
        PARAM.SVM_Type, PARAM.SVM_C, PARAM.SVM_bias, PARAM.SVM_PosW, ...
        PARAM.CategOfInterest, Epoch_Callback );
elseif strcmp( PARAM.Classifier_Type, 'svm-struct' )
    bag_filter_func = @(c,k) deal( 1:size(bestIoU{k},2), bestGtIdx{k}(c,:), bestIoU{k}(c,:) );

    loss_curve_func = @(a) a;

    [classifier_model_all, ~,~] = svm_str_loss_grad_multiclass( ...
            Cache_Pos, ImIds_Pos, imageN, feat_scale_factor, ... 
            bag_func, bag_filter_func, PARAM.Train_Pos_IoU_Threshold, PARAM.Train_Neg_IoU_Threshold, ... 
            500, 2000, PARAM.SVM_Epoch, PARAM.SVM_C, PARAM.SVM_bias, PARAM.SVM_PosW, ... 
             PARAM.SVM_ConstType, PARAM.SVM_HingeLossType, loss_curve_func);

%{
   classifier_model_all = svm_str_loss_grad( Cache_Pos, ImIds_Pos, imageN, 1, ... 
           bag_func, bag_filter_func, PARAM.Train_Pos_IoU_Threshold, PARAM.Train_Neg_IoU_Threshold, ... 
           500, 2000, PARAM.SVM_Epoch, PARAM.SVM_C2, PARAM.SVM_bias, C1, PARAM.CategOfInterest);
%}

%{
   classifier_model_all = SVMLoc_HardTrain( Cache_Pos, ImIds_Pos, imageN, 1, ...
        bag_func, bag_filter_func, PARAM.Train_Pos_IoU_Threshold, PARAM.Train_Neg_IoU_Threshold, ...
        500, 5000, PARAM.SVM_Epoch, PARAM.SVM_C, PARAM.SVM_bias, PARAM.SVM_PosW, ...
        loss_curve_func1, PARAM.SVMLoc_Neg_Organization, use_fixed_C, ...
        misc_param, PARAM.CategOfInterest, Epoch_Callback);
%}
else
    error( 'Unrecognized Classifier_Type' );
end

classifier_model_all.scale = feat_scale_vec.';

%% Save classifiers

SaveClassifier( classifier_model_all, PARAM.Classifier_Type, ...
    SPECIFIC_DIRS.Train, PARAM.CategOfInterest );


fprintf( 1, 'Done\n' );


