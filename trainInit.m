function [PARAM, SPECIFIC_DIRS, CategIdOfInterest] = trainInit( model_type )
% TRAININIT initilize all the parameters for training
% 
% Usage:
%
%   [PARAM, SPECIFIC_DIRS, CategIdOfInterest] = trainInit( model_type )
%
% Input:
%
%   model_type: can be a string naming the classifier model type:
%       'struct' - (linear) structured SVM, 
%                  use the models in ./models_svm_linear
%       'linear' - ordinary linear SVM
%                  use the models in ./models_svm_struct
%
% Output: All outputs are useful for training
%
%   PARAM: all the parameters for training
%     They are specified in 
%       ./train_pipeline/params_commons.m
%       ./train_pipeline/params_svm_linear.m
%       ./train_pipeline/params_svm_struct.m
%
%   SPECIFIC_DIRS: cache paths for all the trainging stages
%     Training with linear and structured SVM shared some stages (used the
%     same cache paths), and differs at other stages (used different cache paths)
%    Also refer to the ./train_pipeline/params_*.m
%
%   SPECIFIC_DIRS: is a vector of integer number that indicates
%     which categories should be included in the further training
%    It is always be 1:N, where N is the number of categories.

% addpath

TOOLBOX_ROOT_DIR = fileparts(which(mfilename('fullpath')));
addpath( genpath( fullfile( TOOLBOX_ROOT_DIR, 'dependency' ) ) );
addpath( genpath( fullfile( TOOLBOX_ROOT_DIR, 'caffe/matlab/caffe' ) ) );
addpath( genpath( fullfile( TOOLBOX_ROOT_DIR, 'train_pipeline' ) ) );
addpath( genpath( fullfile( TOOLBOX_ROOT_DIR, 'voc2007/VOCdevkit/VOCcode' ) ) );
addpath( TOOLBOX_ROOT_DIR );

% compile
compile_dependency

% settings
PARAM = struct();
SPECIFIC_DIRS = struct();
params_common;  % load common settings

switch model_type
    case 'linear'
        params_svm_linear;  % load specific settings for linear SVM
    case 'struct'
        params_svm_struct;  % load specific settings for structured SVM
    otherwise
        error( 'Unrecognized svm_type' );
end

% cache path
CACHE_ROOT = fullfile( TOOLBOX_ROOT_DIR, 'voc2007_train_cache' );
specific_fieldnames = fieldnames(SPECIFIC_DIRS);
for k = 1:length(specific_fieldnames)
    cmdstr = sprintf('SPECIFIC_DIRS.%s = fullfile(CACHE_ROOT,SPECIFIC_DIRS.%s);', ...
        specific_fieldnames{k}, specific_fieldnames{k});
    eval( cmdstr );
    mkdir_p( eval(sprintf('SPECIFIC_DIRS.%s',specific_fieldnames{k})) );
end
SPECIFIC_DIRS.VOC2007_ROOT = fullfile( TOOLBOX_ROOT_DIR, 'voc2007/VOCdevkit/VOC2007' );

%
CategIdOfInterest = PARAM.CategIdOfInterest;
