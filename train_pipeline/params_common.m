
%% set up parameters

PARAM.CategOfInterest =...
        {'aeroplane','bicycle','bird','boat','bottle','bus','car','cat',...
        'chair','cow','diningtable','dog','horse','motorbike','person',...
        'pottedplant','sheep','sofa','train','tvmonitor'};
PARAM.CategIdOfInterest = 1:length(PARAM.CategOfInterest);
PARAM.DataSubSet_Names = {'trainval', 'test'};

PARAM.Patch_Padding  = 16;

PARAM.Feature_Norm    = 20;
PARAM.Train_DataSet   = 'trainval';
% Classifier_Type : 'svm-linear', 'svm-localization'
PARAM.Train_Neg_IoU_Threshold = 0.3;

PARAM.Test_DataSet  = 'test';

PARAM.Truncate_BBox2Boundary  = 1;

PARAM.NMS_Threshold           = 0.3;
PARAM.Detection_IoU_Threshold = [ 0.5 0.7 ];
PARAM.MaxKeptBoxPerImage      = 400;
PARAM.MaxKeptObjectPerImage   = 40;

PARAM.Test_BBoxReg_Enabled    = 0;
PARAM.Test_BBoxReg_Param      = struct();

PARAM.Test_AP_Method = 'voc2007';

% GP Train region
PARAM.GP_ADDITIONAL_REGION = {};
PARAM.GP_ADDITIONAL_REGION{1}.GTNeighbor_Num = 50;
PARAM.GP_ADDITIONAL_REGION{1}.GTNeighbor_MaxCenterShift = 0.3;
PARAM.GP_ADDITIONAL_REGION{1}.GTNeighbor_MaxScaleShift = sqrt(2);

PARAM.GP_ADDITIONAL_REGION{2}.GTNeighbor_Num = 20;
PARAM.GP_ADDITIONAL_REGION{2}.GTNeighbor_MaxCenterShift = 0.15;
PARAM.GP_ADDITIONAL_REGION{2}.GTNeighbor_MaxScaleShift = sqrt(sqrt(2));


PARAM.GP_Local_IoU_Threshold   = 0.3;
PARAM.GP_Local_MaxSampleNum    = inf;  % inf - use all available bboxes around a groundtruth 
PARAM.GP_Train_MaxLocalityNum  = 10000; % limited to a certain number for efficiency reason
PARAM.GP_Train_MaxIterationNum = 200;
PARAM.GP_MeanFunc = '@meanConst';
PARAM.GP_CovFunc  = '@covSEard';
PARAM.GP_LikFunc  = '@likGauss';
PARAM.GP_BBoxParamType = 'yxhwl';

PARAM.GP_Sequential_MaxIterationNum = 16;
PARAM.GP_Sequential_GapIterationNum = 4;
PARAM.GP_Sequential_MaxLocalityNumPerImage = inf;
PARAM.GP_Sequential_ScoreThreshold  = -1;
PARAM.GP_Sequential_Local_IoU_Threshold = 0.3;
PARAM.GP_Sequential_NMS_Threshold = 0.3;
PARAM.GP_Orphan_Tag = struct();

PARAM.SGP_Solver = 'minFunc:lbfgs'; % 'minFunc:*', 'fminunc:quasi-newton', 'fminunc:trust-region'

% generate bbox list
PARAM.KeepOriginalLabel = 1; %
PARAM.BBoxRegression_IoUThreshold = 0.6;
PARAM.BBoxRegression_Ridge_Lambda = 1000;

%% Set up cache dirs
SPECIFIC_DIRS.PrepDataset      = 'PrepDataset';
SPECIFIC_DIRS.RegionProposal   = 'RegionProposal';
SPECIFIC_DIRS.BestIoU4Train    = 'BestIoU4Train';
SPECIFIC_DIRS.BoxList4Finetune = 'BoxList4Finetune';
SPECIFIC_DIRS.CaffeModel       = 'CaffeModel';
SPECIFIC_DIRS.Features4Groundtruth = 'Features4Groundtruth/cls';
SPECIFIC_DIRS.Features4Proposed    = 'Features4Proposed/cls';
SPECIFIC_DIRS.Features4Proposed_bboxreg = 'Features4Proposed/bboxreg';
SPECIFIC_DIRS.FeatureNorm4Train         = 'FeatureNorm4Train/cls';
SPECIFIC_DIRS.FeatureNorm4Train_bboxreg = 'FeatureNorm4Train/bboxreg';

SPECIFIC_DIRS.BBoxRegTrain     = 'BBoxRegTrain';

SPECIFIC_DIRS.RegionProposal4GPTrain = 'RegionProposal4GPTrain';
SPECIFIC_DIRS.Features4GPTrain       = 'Features4GPTrain';

