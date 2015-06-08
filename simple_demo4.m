function simple_demo4( image_or_filename, model_type, force_init )
% SIMPLE_DEMO4 detects and shows objects on a single image using a specific classifier (linear/structures SVM). 
%   The detection results without and with GP-based FGS are shown in two
%   figures, respectively.
%   Remark: the detection model is automatically loaded and initialized.
%
% Usage: 
%
%   simple_demo4( image_or_filename, model_type, force_init )
%
% Input:
%
%   image_or_filename: can be either an image matrix loaded by imread(filename ...)
%       or the path of an image file
%
%   model_type: can be a string naming the classifier model type:
%       'struct' - (linear) structured SVM, 
%                  use the models in ./models_svm_linear
%       'linear' - ordinary linear SVM
%                  use the models in ./models_svm_struct
%   force_init: can be either 0 or 1
%       0 - the detection model is only loaded and initialized when it is
%           used for the first time
%       1 - the detection model is force reloaded (useful when the old model 
%           is replaced by a new model, and you do not want to restart MATLAB)
%

if ischar(image_or_filename)
    I = imread( image_or_filename );
else
    I = image_or_filename;
end

persistent det_model
if isempty(det_model) || ...
        (exist('force_init','var') && force_init) || ...
        (exist('model_type','var') && ~strcmp(det_model.type_flags, model_type) )
    if ~exist( 'model_type', 'var' ) || isempty(model_type)
        model_type = 'struct';
    end

    det_model = detInit( [], [], ['models_svm_' model_type] );
    det_model.type_flags = model_type;
end

% detection without GP refinement
fprintf( 'Run for the case WITHOUT GP\n' );
[Bs,Ss] = detSingle( I, det_model, 0 );
figure
detShowBBoxes( I, Bs, Ss, det_model, 0 ); title( 'Finetuned model without GP searching' ); 
set(gcf,'Color','white');

% detection with GP refinement
fprintf( 'Run for the case WITH GP\n' );
[Bs,Ss] = detSingle( I, det_model, 1 );
figure
detShowBBoxes( I, Bs, Ss, det_model, 0 ); title( 'Finetuned model WITH GP searching' ); 
set(gcf,'Color','white');

end
