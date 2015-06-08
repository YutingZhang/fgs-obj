function [ chosenBs, chosenSs ] = boxes_and_scores_after_nms( ...
    boxes, scores, NMS_Threshold, MaxKeptBoxPerImage, MaxKeptObjectPerImage )
% support both cell array and a single input

if ~exist('MaxKeptBoxPerImage','var') || isempty(MaxKeptBoxPerImage)
    MaxKeptBoxPerImage = inf;
end

if ~exist('MaxKeptObjectPerImage','var') || isempty(MaxKeptObjectPerImage)
    MaxKeptObjectPerImage = inf;
end

if iscell( boxes )
    
    imageN = length(boxes);
    
    chosenBs = cell(imageN,1);
    chosenSs = cell(imageN,1);
    for k = 1:imageN
        [ chosenBs{k}, chosenSs{k} ] = boxes_and_scores_after_nms_single( ...
            boxes{k}, scores{k}, NMS_Threshold, MaxKeptBoxPerImage, MaxKeptObjectPerImage );
    end

else
    
    [ chosenBs, chosenSs ] = boxes_and_scores_after_nms_single( ...
            boxes, scores, NMS_Threshold, MaxKeptBoxPerImage, MaxKeptObjectPerImage );
    
end

end

function [ chosenB, chosenS ] = boxes_and_scores_after_nms_single( ...
    boxes, scores, NMS_Threshold, MaxKeptBoxPerImage, MaxKeptObjectPerImage )

chosenIdxB = firstKindexB( scores, MaxKeptBoxPerImage, 'descend' );
% nms
furtherChosenIdx  = nms( [boxes(chosenIdxB,:), ...
        scores(chosenIdxB)], NMS_Threshold, 'iou' );
furtherChosenIdxB = false( sum(chosenIdxB),1 );
furtherChosenIdxB( furtherChosenIdx ) = true;
chosenIdxB( chosenIdxB ) = furtherChosenIdxB;

chosenS = scores(chosenIdxB);
chosenB = boxes(chosenIdxB,:);
[chosenS, sortedIdx] = sort( chosenS, 'descend' ); % sort in decrease order (consistent with VOC)
chosenB = chosenB(sortedIdx,:);

chosenS = chosenS(1:min(end,MaxKeptObjectPerImage));
chosenB = chosenB(1:min(end,MaxKeptObjectPerImage),:);

end
