function det_model = detInit( caffe_use_gpu, caffe_batch_size, model_dir )
% DETINIT loads and initializes the detection model
%   paths for dependency is also added
% 
% Intro:
%   The detection model consists of the following components:
%     - the CNN feature extraction model
%     - the linear/structured SVM classifier
%     - the bounding box regression model
%     - the GP-based FGS model
%
% Usage:
%
%   det_model = detInit( caffe_use_gpu, caffe_batch_size, model_dir )
% 
% Input:
%
%   caffe_use_gpu: can be 0 or 1 (default) to indicate whether the Caffe 
%       toolbox should use CPU or GPU
%   
%   caffe_batch_size: can be an integer >=1
%       Default value: 32
%       Caffe toolbox process input in batch. A proper batch number can get
%       a good balance between the efficiency and memory consumption
%
%   model_dir: specify the root directory of the detection model
%       Default value: ./models_svm_struct
%                   (The trained structured SVM model)
%


%% init param

TOOLBOX_ROOT_DIR = fileparts(which(mfilename('fullpath')));
addpath( genpath( fullfile( TOOLBOX_ROOT_DIR, 'dependency' ) ) );
addpath( genpath( fullfile( TOOLBOX_ROOT_DIR, 'caffe/matlab/caffe' ) ) );
addpath( TOOLBOX_ROOT_DIR );

% ---------------------
compile_dependency
% --------------------


if ~exist( 'caffe_use_gpu', 'var' ) || isempty(caffe_use_gpu)
    caffe_use_gpu = 1;
end

if ~exist( 'caffe_batch_size', 'var' ) || isempty(caffe_batch_size)
   caffe_batch_size = 32;
end

if ~exist( 'model_dir', 'var' ) || isempty(model_dir)
   model_dir = fullfile( fileparts( which( mfilename('fullpath') ) ), 'models_svm_struct' );
end

det_model = struct();
det_model.model_dir = model_dir;

%% load classification model for 20 category, i.e., CNN network

if caffe_use_gpu<0
    fprintf( 'WARNING: CAFFE is not set up, you can only run the cached demo for VOC2007' );
else

% check whether matlab wrapper is there
if exist( 'caffe','file' ) ~= 3
    error( 'FATAL: caffe matlab wrapper was not set up' );
end

try
    if caffe('get_version') ~= 20140912
        error( 'FATAL: wrong version for caffe matlab wrapper' );
    end
catch
    error( 'FATAL: please use the customized matlab wrapper' );
end

% initialize caffe model
matcaffe_init1( fullfile(det_model.model_dir, 'cnn/deploy.prototxt'), ...
    fullfile(det_model.model_dir, 'cnn/caffe_model'), caffe_use_gpu, caffe_batch_size );
% figure out the canonical patch size
caffe_input_response_ids = caffe( 'get_input_response_ids' );
caffe_response_info = caffe('get_response_info');
canonical_patchsize = caffe_response_info(caffe_input_response_ids).size;
canonical_patchsize = canonical_patchsize([2 1 3]);
% load input mean
MAT_CONTENT = load( fullfile(det_model.model_dir, 'cnn/input_mean.mat') );
caffe_input_mean = MAT_CONTENT.image_mean;
clear MAT_CONTENT;
% set up the feat_func (to classification layer)
feat_func = @(patches) matcaffe_run_wrapper( single(patches), caffe_input_mean, {'fc7','pool5'} );

det_model.cnn.feat_func = feat_func;
det_model.cnn.canonical_patchsize = canonical_patchsize;
det_model.cnn.padding       = 16;
det_model.cnn.batch_size    = caffe_batch_size;
det_model.cnn.max_batch_num = 5;

end

%% load category list
MAT_CONTENT = load( fullfile(det_model.model_dir, 'categ_list.mat') );
det_model.categ_list = MAT_CONTENT.CategList;
clear MAT_CONTENT

%% load gp model
det_model.gp = cell(length(det_model.categ_list),1);
for c = 1:length(det_model.categ_list)
    det_model.gp{c} = load( fullfile(det_model.model_dir, 'gp', [det_model.categ_list{c} '.mat']) );
end
det_model.gp = cell2mat(det_model.gp);

%% load regression model for 20 categories
det_model.bboxreg = GetRegressor( det_model.categ_list, fullfile(det_model.model_dir, 'bboxreg') ) ;

%% load classifier model for 20 categories
det_model.classifier = GetClassifier( det_model.categ_list, fullfile(det_model.model_dir, 'classifier') ) ;

%% initiaize region proposal model

SelectiveSearchInit();
rp_func = @(im) SelectiveSearchOnOneImage( im, 'ijcv_fast' );

det_model.rp_func = rp_func;

