function [bestIoU, bestGtIdx] = BestIoU_nonCell( boxes, gtBoxes )

N1 = size( boxes , 1 );

bestIoU   = zeros( N1,1 );
bestGtIdx = zeros( N1,1 );

if isempty( gtBoxes )
    % bestIoU(:) = 0;  % already 0 ...
else
    S = PairedIoU( boxes, gtBoxes ).';
    [bestIoU(:), bestGtIdx(:)] = max( S, [], 1 );
	bestGtIdx(~bestIoU) = 0;
end

end

