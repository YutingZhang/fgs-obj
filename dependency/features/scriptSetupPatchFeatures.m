if isfield( PARAM, 'Feature_BatchSize' )
    batchsize_for_feature_extractor = PARAM.Feature_BatchSize;
else
    batchsize_for_feature_extractor = 128;
end
feat_func = @(patches) error('No feature function is specified');
canonical_patchsize = [nan nan nan];

% initialize caffe model
caffe_batch_size = batchsize_for_feature_extractor;
caffe_use_gpu = 1;
if isfield( PARAM, 'Caffe_UseGPU' )
    caffe_use_gpu = PARAM.Caffe_UseGPU;
end
matcaffe_init1( fullfile(SPECIFIC_DIRS.CaffeModel, 'deploy.prototxt'), ...
    fullfile(SPECIFIC_DIRS.CaffeModel, 'caffe_model'), caffe_use_gpu, caffe_batch_size );
% figure out the canonical patch size
caffe_input_response_ids = caffe( 'get_input_response_ids' );
caffe_response_info = caffe('get_response_info');
canonical_patchsize = caffe_response_info(caffe_input_response_ids).size;
canonical_patchsize = canonical_patchsize([2 1 3]);
% load input mean
MAT_CONTENT = load( fullfile(SPECIFIC_DIRS.CaffeModel, 'input_mean.mat') );
caffe_input_mean = MAT_CONTENT.image_mean;
clear MAT_CONTENT;
% set up the feat_func
feat_func = @(patches) matcaffe_run_wrapper( single(patches), caffe_input_mean, Caffe_Layer );
