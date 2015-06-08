function benchmark_voc2007( model_type, gp_enabled )
% BENCHMARK_VOC2007 do benchmark test on PASCAL VOC2007 test set using the trained detection models
% 
% Usage:
%   BENCHMARK_VOC2007( MODEL_TYPE, GP_ENABLED ) benchmark using specific
%   classifier model with/without the Gaussian process (GP) based fine-
%   grained target search (FGS)
%
%   MODEL_TYPE can be a string naming the classifier model type:
%       'struct' - (linear) structured SVM, 
%                  use the models in ./models_svm_linear
%       'linear' - ordinary linear SVM
%                  use the models in ./models_svm_struct
%
%   GP_ENABLED can be 0 or 1 (default)
%       0 - Only selective search is used to propose regions
%       1 - GP-based FGS will be applied afterward.
%


if ~exist( 'model_type', 'var' ) || isempty(model_type)
    model_type = 'struct';
end

if ~exist('gp_enabled','var') || isempty(gp_enabled)
    gp_enabled = 1;
end

detInitPath;

if exist('caffe','file') == 3
    [RECALL,PREC,AP,categ_list]=detVOC2007(model_type, gp_enabled, 1);
else
    [RECALL,PREC,AP,categ_list]=detVOC2007(model_type, gp_enabled,-1);
end

gp_flags_str = { 'with-gp', 'no-gp' };
SUMMARY_FILENAME = fullfile('voc2007_results_cache', ...
    [model_type '_' gp_flags_str{2-gp_enabled} '.mat']);

save( SUMMARY_FILENAME, 'RECALL', 'PREC', 'AP', 'categ_list' );
fprintf( 'Results are saved to %s\n', SUMMARY_FILENAME );

cellfun( @(t,a) fprintf('%10s : \t%.2f%%\n',t,a*100), ...
    [vec(categ_list);'mAP'], num2cell([AP;mean(AP)]) );

clf
for c = 1:length(categ_list)
    subplot( 4, ceil(length(categ_list)/4), c );
    plot( RECALL{c}, PREC{c} );
    xlabel( 'recall' );
    ylabel( 'prec' );
    title( sprintf('%s - AP: %f', categ_list{c}, AP(c) ) );
end

end
