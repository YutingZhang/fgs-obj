function [Fcls, Freg, bboxes] = detSingleInitialFeatures( I, det_model, bboxes )
% DETSINGLEINITIALFEATURES extract features from the initial bounding boxes on a single image.
%
% Usage:
%
%   I: is an image matrix loaded by imread (e.g., I = imread('000220.jpg');).
%
% Input:
%
%   det_model: is the detection model loaded by detInit(...)
%
%   bboxes: can be M*4 maxtrix for intial bounding box coordinates, where 
%     M is the number of initial bounding boxes. Each row should be in the
%     form of [ymin, xmin, ymax, xmax].
%   If bboxes is not specified or empty, Selective Search toolbox is called
%   to generated bounding box proposals
%
% Output:
%
%   Fcls: features for classification
%
%   Freg: features for bounding box regression
%
%   bboxes: refer to the input
%


% region proposal (Selective Search)
fprintf('Region proposal : '); tic
if ~exist('bboxes','var') || isempty(bboxes)
    bboxes = det_model.rp_func( I );
    toc
else
    fprintf( 'use cached\n' );
end

% extract features
fprintf('Feature extraction : %d boxes : ', size(bboxes,1) ); tic

F = features_from_bboxes( I, bboxes, ...
    det_model.cnn.canonical_patchsize, ...
    det_model.cnn.padding, det_model.cnn.feat_func, ...
    det_model.cnn.max_batch_num * det_model.cnn.batch_size );

Fcls = F{1}; Freg = F{2};
toc

end
