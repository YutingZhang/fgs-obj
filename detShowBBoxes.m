function detShowBBoxes( I, Bs, Ss, model_or_list, thresh )
% DETSHOWBBOXES visualizes the detection results.
%   Bounding boxes with detection scores are shown on the image.
%   The category label and detection score of a bounding box is shown at
%   the corner of the box. Bounding boxes of each category is in a
%   different color. 
%
% Usage:
%   detShowBBoxes( I, Bs, Ss, model_or_list, thresh )
%
% Input:
%
%   I: an image matrix loaded by imread (e.g., I = imread('000220.jpg');)
%
%   Bs, Ss: the detected bounding boxes and their scores returned by
%      detSingle( I, ... )
%
%   model_or_list: can be either a model initialized by detInit(...) or
%      a list category names, e.g. returned by detVOC2007(...)
%
%   thresh: threshold for pruning false postives, i.e., only boundling
%   boxes with scores higher than thresh will be counted.
%       Default value: -inf
%   Remark: one can set a lower thresh for detSingle(...) to get more
%   bounding boxes, and tune the thresh here to get a good visualization
%   result
%
%

if isstruct( model_or_list )
    categ_list = model_or_list.categ_list;
else
    categ_list = model_or_list;
end

if ~exist('thresh','var') || isempty(thresh)
    thresh = -inf;
end
if isscalar( thresh )
    thresh = repmat(thresh, 1, length( categ_list ));
end

COLOR = colorcube( ceil( length(categ_list) * 2 ) );

clf
imshow(I);
for c = 1:length(categ_list)
    for j = find( reshape(Ss{c},1,length(Ss{c}))>=thresh(c) )
        cur_bbox = num2cell( Bs{c}(j,:) );
        [y1, x1, y2, x2] = deal( cur_bbox{:} );
        lh = line([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1]);
        set( lh, 'Color', COLOR(c,:), 'LineWidth', 2 );
        text(x1,y1, sprintf( '%s: %.2f', categ_list{c}, Ss{c}(j) ), ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', ...
            'Margin', 1, 'BackgroundColor', COLOR(c,:), 'Color', 'white' ) ;
    end
end
