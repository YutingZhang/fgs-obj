function S = PairedIoU( boxes1, boxes2 )
% S = PairedIoU( boxes1, boxes2 )
% boxes1 = M * 4, boxes2 = N * 4
% S = M * N;

if ~exist('boxes2','var') || ~size(boxes2,1)
    boxes2 = boxes1;
end

B1 = boxes1(:,1:4).'; 
B2 = boxes2(:,1:4).'; B2 = reshape( B2, [4, 1, size(B2,2)] );

% interSize = prod( bsxfun( @min, B1([3 4],:,:), B2([3 4],:,:) ) - ...
%     bsxfun( @max, B1([1 2],:,:), B2([1 2],:,:) ) + 1, 1 );

interEdge = min( repmat( B1([3 4],:,:), 1, 1, size(B2,3) ), ...
    repmat( B2([3 4],:,:), 1, size(B1,2), 1 ) ) - ...
    max( repmat( B1([1 2],:,:), 1, 1, size(B2,3) ), ...
    repmat( B2([1 2],:,:), 1, size(B1,2), 1 ) ) + 1;
interEdge = max( interEdge, 0 );
interSize = interEdge(1,:,:).*interEdge(2,:,:);


interSize = reshape( interSize, size(interSize,2), size(interSize,3) );

b1Size = ( B1(3,:)-B1(1,:)+1 ) .* ( B1(4,:)-B1(2,:)+1 );
b2Size = ( B2(3,:)-B2(1,:)+1 ) .* ( B2(4,:)-B2(2,:)+1 );

S = interSize./( bsxfun( @plus, b1Size.', b2Size ) - interSize );


 
