function [Bs, Ss] = detSingle( I, det_model, gp_enabled, thresh, gp_thresh, ...
    bboxes, CategOfInterest4GP )
% DETSINGLE detects objects on a single image using an existing detection model
%
% Usage:
%   [Bs, Ss] = detSingle( I, det_model, gp_enabled, thresh, gp_thresh, ...
%       bboxes, CategOfInterest4GP )
% 
% INPUT:
%
%   I: is an image matrix loaded by imread (e.g., I = imread('000220.jpg');).
%
%   det_model: is the detection model loaded by detInit(...)
%
%   gp_enabled: can be 0 or 1 (default)
%       0 - Only selective search is used to propose regions
%       1 - GP-based FGS will be applied afterward.
%
%   thresh: threshold for pruning false postives, i.e., only boundling
%   boxes with scores higher than thresh will be counted.
%       Default value: 0
%
%   gp_thresh: threshold for candidate region for the GP-based FGS.
%       Default value: -1
%
%   bboxes: can be M*4 maxtrix for intial bounding box coordinates, where 
%     M is the number of initial bounding boxes. Each row should be in the
%     form of [ymin, xmin, ymax, xmax].
%   If bboxes is not specified or empty, Selective Search toolbox is called
%   to generated bounding box proposals
%
%   CategOfInterest4GP: can be a string cell array indicating which
%     categories FGS should be applied. It is only useful when gp_enable==1
%     By default FGS is applied to all the categories.
%     E.g. CategOfInterest4GP = {'aeroplane','cow'}
%     Remark: Whatever CategOfInterest4GP is specified, detection is done
%     for all the categories.
%
% OUTPUT: Let N be the number of categories
%
%   Bs : N-d cell array. Bs{i} is the coordinates of the detected bounding 
%     boxes for the i-th category, which is a matrix with 4 columns. 
%     Refer to "bboxes" (INPUT) for the format.
%
%   Ss : N-d cell array. Ss{i} is a vector of the scores of the detected
%     bounding boxes for the i-th category.
%

if ~exist('thresh','var') || isempty(thresh)
    thresh = -0;
end

if ~exist('gp_thresh','var') || isempty(gp_thresh)
    gp_thresh = -1;
end

if ~exist('bboxes','var')
    bboxes = [];
end

if ~exist('CategOfInterest4GP','var') || isempty(CategOfInterest4GP)
    CategIdOfInterest4GP = [];
else
    [~,CategIdOfInterest4GP] = ismember( CategOfInterest4GP, det_model.categ_list );
end

t0=tic;

persistent I0 Fcls0 Freg0 bboxes0

if size(I,3)==1
    I = repmat( I, 1, 1, 3 );
end

if ~iscell(bboxes)
    if ~isequal(I0,I)
        [Fcls0, Freg0, bboxes0] = detSingleInitialFeatures( I, det_model, bboxes );
        I0 = I;
    end
else
    bboxes0 = bboxes{1};
    Fcls0   = bboxes{2};
    Freg0   = bboxes{3};
end

% apply classifier
fprintf('Apply classifier : '); tic
S0 = ApplyClassifier(Fcls0, det_model.classifier);
toc

% bbox regression
fprintf('BBox regression : '); tic
B1 = ApplyBBoxRegressor( bboxes0, Freg0, det_model.bboxreg );
B1 = mat2cell( B1, size(B1,1), size(B1,2), ones(size(B1,3),1) );

S1 = cell(length( det_model.categ_list ),1);
for c = 1:length( det_model.categ_list )
    S1{c} = S0(c,:).';
end

toc

% gp searching
if gp_enabled
    fprintf( 'GP searching: ' ); tic
    [newB, newS] = detSingleGP(I,bboxes0,S0,det_model,gp_thresh,CategIdOfInterest4GP);
    for c = 1:length( det_model.categ_list )
        if ~isempty( newB{c} )
            newFreg_c = features_from_bboxes( I, newB{c}, ...
                det_model.cnn.canonical_patchsize, ...
                det_model.cnn.padding, det_model.cnn.feat_func, ...
                det_model.cnn.max_batch_num * det_model.cnn.batch_size );
            newB1c = ApplyBBoxRegressor( newB{c}, newFreg_c{2}, det_model.bboxreg );
            B1{c} = [B1{c};newB1c(:,:,c)];
            S1{c} = [S1{c};newS{c}];
        end
    end
    fprintf( ' ' );
    toc
end

% nms & thresh
fprintf('NMS : '); tic

if isscalar( thresh )
    thresh = repmat(thresh, 1, length( det_model.categ_list ));
end

Bs = cell(length( det_model.categ_list ),1);
Ss = cell(length( det_model.categ_list ),1);

for c = 1:length( det_model.categ_list )
    
    B1{c}(:,1) = max( B1{c}(:,1), 1 );
    B1{c}(:,2) = max( B1{c}(:,2), 1 );
    B1{c}(:,3) = min( B1{c}(:,3), size(I,1) );
    B1{c}(:,4) = min( B1{c}(:,4), size(I,2) );
    
    [Bs{c}, Ss{c}] = boxes_and_scores_after_nms( ...
        B1{c}, S1{c}, 0.3, [], [] );
    chosenIdxB = Ss{c}<thresh(c);
    Ss{c}(chosenIdxB)   = [];
    Bs{c}(chosenIdxB,:) = [];
end
toc

fprintf( 'Total elapsed time : ' );
toc(t0);

