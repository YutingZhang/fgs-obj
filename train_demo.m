function train_demo( model_type )
% TRAIN_DEMO trains the detection model from scratch (except for the CNN model training)
%   CNN model training can be done by ./finetune_vgg16_on_voc2007.sh
% 
% Usage:
%
%   train_demo( model_type )
%
% Input: 
%
%   model_type: can be a string naming the classifier model type:
%       'struct' - (linear) structured SVM, 
%                  use the models in ./models_svm_linear
%       'linear' - ordinary linear SVM
%                  use the models in ./models_svm_struct
%
% Output:
%
%   The trained (intermediate) models and intermediate outputs are all in
%     the cache folders (SPECIFIC_DIRS in the code)
%

if ~exist('model_type','var') || isempty( model_type )
    model_type = 'struct';
end

%% Preparation

fprintf( '========== Preparation\n' );

switch model_type
    case 'struct'   % do initialization for structured svm model
        trainInit_svmStruct;
    case 'linear'   % do initialization for linear svm model
        trainInit_svmLinear
    otherwise
        error( 'Unknown model_type' );
end

% prepare dataset
trainCallStage('PrepDataset');

% do selective search (on trainval & test) 
trainCallStage('RegionProposal');

%% Classifier & BBox regression model training

fprintf( '========== Classifier & BBox regression model training\n' );

% compute best iou with gt (on trainval)
trainCallStage('BestIoU4Train');

% extract features from proposed bboxes (on trainval & test)
trainCallStage('Features4Proposed');

% extract features from groundtruth bboxes (on trainval)
trainCallStage('Features4Groundtruth');

% compute mean feature norm (on trainval)
trainCallStage('FeatureNorm4Train');

% train classifier
trainCallStage('Train');

% train bbox regression model
trainCallStage('BBoxRegTrain');

%% GP model training

fprintf( '========== GP model training\n' );

% generate additional regions for GP training
trainCallStage('RegionProposal4GPTrain');

% extract features from the additional regions
trainCallStage('Features4GPTrain');

% test on the training set (observations for GP training)
trainCallStage('Test4GPTrain');

% gp training
trainCallStage('GPTrain');


