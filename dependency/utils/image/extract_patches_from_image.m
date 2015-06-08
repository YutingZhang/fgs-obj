function P = extract_patches_from_image( I, boxes, canonical_patchsize, padding, paddingValue )
% P = extract_patches_from_image( I, boxes, canonical_patchsize [, padding, paddingValue] )
% Remark: boxes = N*[y1 x1 y2 x2] or = N*[y1 x1 y2 x2 flip_x]
%  padding = [t r b l] (same as CSS)

canonical_channelNum = [];  % keep unchanged
if numel(canonical_patchsize) == 1
    canonical_patchsize = [canonical_patchsize, canonical_patchsize];
end
if numel(canonical_patchsize) == 3
    canonical_channelNum = canonical_patchsize(3);
    canonical_patchsize  = canonical_patchsize(1:2);
end

if ~isempty(canonical_channelNum)
    switch canonical_channelNum
        case 1
            switch size(I,3)
                case 1
                case 3
                    I = rgb2gray(I);
                otherwise
                    error( 'Unsupported conversion' );
            end
        case 3
            switch size(I,3)
                case 1
                    I = repmat( I, [1,1,3] );
                case 3
                otherwise
                    error( 'Unsupported conversion' );
            end
        otherwise
            error('Unsupported channel number');
    end 
end

if ~exist( 'padding', 'var' ) || isempty(padding)
    padding = 0;
end
if numel(padding)==1
    padding = [padding(1), padding(1)];
end
if numel(padding)==2
    padding(3) = padding(1);
end
if numel(padding)==3
    padding(4) = padding(2);
end
padding = reshape( padding,1,numel(padding) );

if ~exist( 'paddingValue', 'var' ) || isempty(paddingValue)
    paddingValue = nan;
end
paddingValue = eval( [class(I) '(paddingValue)' ] );


if size(boxes,2)>4
    FLIP_X = boxes(:,5);
    boxes = boxes(:,1:4);
else
    FLIP_X = zeros(size(boxes,1),1);
end

center_patchsize = canonical_patchsize-padding([1 4])-padding([3 2]);

spadding = [-padding([1 4]),padding([3 2])];    % convert to [-t -l b r]
sscale = spadding./[center_patchsize,center_patchsize]*2 + [-1 -1 1 1];

bH0h = (boxes(:,3)-boxes(:,1)+1)/2;
bW0h = (boxes(:,4)-boxes(:,2)+1)/2;
bCY0 = boxes(:,1)+bH0h;
bCX0 = boxes(:,2)+bW0h;

boxes = round(bsxfun(@times,[bH0h,bW0h,bH0h,bW0h], sscale)+[bCY0,bCX0,bCY0,bCX0]);

iHW = [ size(I,1), size(I,2) ];

bH =  boxes(:,3)-boxes(:,1)+1;
bW =  boxes(:,4)-boxes(:,2)+1;


TL = bsxfun(@max,boxes(:,[1 2]), [1 1]);
BR = bsxfun(@min,boxes(:,[3 4]), iHW );
sH = (BR(:,1)-TL(:,1)+1)./bH * canonical_patchsize(1);
sW = (BR(:,2)-TL(:,2)+1)./bW * canonical_patchsize(2);
sH = floor(sH+0.5); sW = floor(sW+0.5);

JT_1 = max( 1-boxes(:,1), 0 )./bH * canonical_patchsize(1);
JL_1 = max( 1-boxes(:,2), 0 )./bW * canonical_patchsize(2);
JT_1 = ceil(JT_1-0.5); JL_1 = ceil(JL_1-0.5);

m = size(boxes,1);
P = repmat( paddingValue, [canonical_patchsize, size(I,3), m ] );

for k = 1:m
    J = I(TL(k,1):BR(k,1),TL(k,2):BR(k,2),:);
    if numel(J) && (sH(k)*sW(k))
        P(JT_1(k)+(1:sH(k)),JL_1(k)+(1:sW(k)),:,k) = imresize( J, [sH(k) sW(k)], 'bilinear', 'antialiasing', false );
        if FLIP_X(k)
            P(:,:,:,k) = P(:,end:-1:1,:,k);
        end
    end
end

end
