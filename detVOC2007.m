function [RECALL, PREC, AP, categ_list] = detVOC2007(...
    model_type, gp_enabled, use_gpu, CategOfInterest4GP )
% DETVOC2007 is the internal function for benchmarking the existing detection models on PASCAL VOC2007 test set
%
% Usage:
%   [RECALL, PREC, AP, categ_list]  = detVOC2007( model_type, gp_enabled, use_gpu, CategOfInterest4GP )
%   benchmark using specific classifier model with/without the Gaussian 
%   process (GP) based fine-grained target search (FGS). More options can
%   be specified.
%
%  INPUT:
%   model_type: can be a string naming the classifier model type:
%       'struct' - (linear) structured SVM, 
%                  use the models in ./models_svm_linear
%       'linear' - ordinary linear SVM
%                  use the models in ./models_svm_struct
%
%   gp_enabled: can be 0 or 1 (default)
%       0 - Only selective search is used to propose regions
%       1 - GP-based FGS will be applied afterward.
%
%   use_gpu: can be 0 or 1 (default)
%       0 - the Caffe toolbox will run on CPU only.
%       1 - the Caffe toolbox will mainly run on GPU
%     Remark: GPU device id can be specified by caffe('set_device',gpu_id)
%
%   CategOfInterest4GP: can be a string cell array indicating which
%     categories FGS should be applied. It is only useful when gp_enable==1
%     By default FGS is applied to all the categories.
%     E.g. CategOfInterest4GP = {'aeroplane','cow'}
%     Remark: Whatever CategOfInterest4GP is specified, detection is done
%     for all the categories.
%
%  OUTPUT: Let N be the number of categories
%   RECALL, REC : N-d cell arrays. RECALL{i},REC{i} together form the precision-recall curve for the i-th category  
%   AP          : a N-d vector of the average precisions of all the categories
%   categ_list  : a N-d string cell array for the categories names
%

%% initial data and models
if ~exist('use_gpu','var') || isempty(use_gpu)
    use_gpu = 1;
end

if ~exist( 'model_type', 'var' ) || isempty(model_type)
    model_type = 'struct';
end
if ~ismember( model_type, {'linear','struct'} )
    error( 'Unknown model_type' );
end

if ~exist('gp_enabled','var') || isempty(gp_enabled)
    gp_enabled = 1;
end

if ~exist('CategOfInterest4GP','var') || isempty(CategOfInterest4GP)
    CategOfInterest4GP = [];
end

fprintf( 'Initialize model: ' ); tic
det_model = detInit(use_gpu,[],['models_svm_' model_type]);
toc

TOOLBOX_ROOT_DIR = fileparts(which(mfilename('fullpath')));
addpath( fullfile( TOOLBOX_ROOT_DIR, 'voc2007/VOCdevkit/VOCcode' )  );

gp_flags_str = { 'with-gp', 'no-gp' };
full_tag = [model_type '_' gp_flags_str{2-gp_enabled}];
RESULT_CACHE_DIR = fullfile(TOOLBOX_ROOT_DIR,'voc2007_results_cache', full_tag );
mkdir_p(RESULT_CACHE_DIR);

FEATURE_CACHE_DIR = fullfile(TOOLBOX_ROOT_DIR,'voc2007_feature_cache');

fprintf( 'Initialize VOC2007 dataset: ' ); tic
VOCinit
toc
% load test set ('val' for development kit)

fprintf( 'Load image list: ' ); tic
[ids,gt]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');
toc

imgN = length(ids);

fprintf( 'Try to load bounding box cache: ' ); tic
if exist( 'voc2007_test_bbox_cache.mat', 'file' )
    load voc2007_test_bbox_cache.mat
    toc
else
    boxes = cell(imgN,1);
    fprintf( 'Cannot find. Ignored\n' );
end

%% run detection for every image

fprintf( '====== Do detection on VOC2007 images =======\n' );

for i = 1:imgN

    fprintf( '%d / %d (%s): ', i, imgN, ids{i} ); tic
    
    resultFn = fullfile( RESULT_CACHE_DIR, [ ids{i} '.mat' ] );
    
    if exist( resultFn, 'file' )
        fprintf( 'existed\n' );
        continue;
    end
    fprintf('\n')
    
    I = imread(sprintf(VOCopts.imgpath,ids{i}));

    try
        Fcls0 = GetProposedFeature( 'test', ids{i}, fullfile( FEATURE_CACHE_DIR, 'cls') );
        Freg0 = GetProposedFeature( 'test', ids{i}, fullfile( FEATURE_CACHE_DIR, 'bboxreg') );
        bboxes0 = [ boxes(i), Fcls0, Freg0 ];
    catch
        bboxes0 = boxes{i};
    end
    
    [Bs, Ss] = detSingle( I, det_model, gp_enabled , -100, -1, bboxes0, CategOfInterest4GP); 
    for c=1:VOCopts.nclasses
        [~,sidx] = sort( Ss{c}, 'descend' );
        sidx = sidx(1:min(40,end));
        Ss{c}= Ss{c}(sidx); 
        Bs{c}= Bs{c}(sidx,:); 
    end 

    save( resultFn, 'Bs', 'Ss' );
    toc
    
end

clear Bs Ss

%% consolidate results

fprintf( '==============================================\n' );
fprintf( 'Consolidate the results: \n' ); tic

fidC = zeros(VOCopts.nclasses,1);
for c=1:VOCopts.nclasses
    cls = VOCopts.classes{c};
    fidC(c)=fopen(sprintf(VOCopts.detrespath,full_tag,cls),'w');
end

for i = 1:imgN
    tic_toc_print( '%d / %d \n', i, imgN );
    resultFn = fullfile( RESULT_CACHE_DIR, [ ids{i} '.mat' ] );
    R = load( resultFn );
    for c=1:VOCopts.nclasses
        for j=1:length(R.Ss{c})
            fprintf(fidC(c),'%s %f %f %f %f %f\n', ...
                    ids{i},R.Ss{c}(j),R.Bs{c}(j,[2 1 4 3]));
        end
    end
end

for c=1:VOCopts.nclasses
    fclose( fidC(c) );
end

toc

%% PR, AP for results

fprintf( 'Compute statistics (PR, AP): \n' );

RECALL = cell(VOCopts.nclasses,1);
PREC   = cell(VOCopts.nclasses,1);
AP     = zeros(VOCopts.nclasses,1);
for c=1:VOCopts.nclasses
    cls = VOCopts.classes{c};
    [RECALL{c},PREC{c},AP(c)]=VOCevaldet(VOCopts,full_tag,cls,false);
end

categ_list = VOCopts.classes;

end

