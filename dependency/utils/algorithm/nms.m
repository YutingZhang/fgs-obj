function [ topIdx, topBoxes ]  = nms(boxes, overlap, method)
% nms(boxes, overlap, method)  (modified version)
% method: 'self-inter' (default), 'best-inter', 'either-inter', 'IoU'
% Non-maximum suppression. (FAST VERSION)
% Greedily select high-scoring detections and skip detections
% that are significantly covered by a previously selected
% detection.
% NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
% but an inner loop has been eliminated to significantly speed it
% up in the case of a large number of boxes

% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if ~exist('method','var')
    method = [];
end
if isempty(method)
    method = 'self-inter';
end

if isempty(boxes)
  topIdx = [];
  topBoxes = zeros(0,4);
  return;
end

y1 = boxes(:,1);
x1 = boxes(:,2);
y2 = boxes(:,3);
x2 = boxes(:,4);
s = boxes(:,5);

area = (x2-x1+1) .* (y2-y1+1);
[~, I] = sort(s);

pick = s*0;
counter = 1;

switch lower(method)
    case 'self-inter'
        while ~isempty(I)
          
          last = length(I);
          i = I(last);  
          pick(counter) = i;
          counter = counter + 1;
          
          xx1 = max(x1(i), x1(I(1:last-1)));
          yy1 = max(y1(i), y1(I(1:last-1)));
          xx2 = min(x2(i), x2(I(1:last-1)));
          yy2 = min(y2(i), y2(I(1:last-1)));
          
          w = max(0.0, xx2-xx1+1);
          h = max(0.0, yy2-yy1+1);
         
          oa = w.*h;
          o  = oa ./ area(I(1:last-1))>overlap;
          
          I([last; find(o>overlap)]) = [];
        end
    case 'best-inter'
        while ~isempty(I)
          
          last = length(I);
          i = I(last);  
          pick(counter) = i;
          counter = counter + 1;
          
          xx1 = max(x1(i), x1(I(1:last-1)));
          yy1 = max(y1(i), y1(I(1:last-1)));
          xx2 = min(x2(i), x2(I(1:last-1)));
          yy2 = min(y2(i), y2(I(1:last-1)));
          
          w = max(0.0, xx2-xx1+1);
          h = max(0.0, yy2-yy1+1);
         
          oa = w.*h;
          o = oa > overlap*area(i);
          
          I([last; find(o>overlap)]) = [];
        end
    case 'either-inter'
        while ~isempty(I)
          
          last = length(I);
          i = I(last);  
          pick(counter) = i;
          counter = counter + 1;
          
          xx1 = max(x1(i), x1(I(1:last-1)));
          yy1 = max(y1(i), y1(I(1:last-1)));
          xx2 = min(x2(i), x2(I(1:last-1)));
          yy2 = min(y2(i), y2(I(1:last-1)));
          
          w = max(0.0, xx2-xx1+1);
          h = max(0.0, yy2-yy1+1);
         
          oa = w.*h;
          oself = oa ./ area(I(1:last-1))>overlap;
          obest = oa > overlap*area(i);
          o = oself | obest; 
          
          I([last; find(o>overlap)]) = [];
        end
    case 'iou'
        while ~isempty(I)
          
          last = length(I);
          i = I(last);  
          pick(counter) = i;
          counter = counter + 1;
          
          o = PascalOverlap( [y1(i),x1(i),y2(i),x2(i)], ...
                  [y1(I(1:last-1)),x1(I(1:last-1)),y2(I(1:last-1)),x2(I(1:last-1))] );
          
          I([last; find(o>overlap)]) = [];
        end
    otherwise
        error('unrecognized method');
end

topIdx = pick(1:(counter-1));
topBoxes = boxes(topIdx,:);
