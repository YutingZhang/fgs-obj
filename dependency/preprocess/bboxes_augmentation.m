function [ augBoxes, augIds ] = bboxes_augmentation( ...
    gtBoxes, gtIds, augFactor, useMirror, varargin )
% Random :
% [ augBoxes, augImIds ] = bboxes_augmentation( gtBoxes, gtImIds, augFactor, useMirror, maxJittering )
% [ augBoxes, augImIds ] = bboxes_augmentation( gtBoxes, gtImIds, augFactor, useMirror, maxCenterShift, maxScaleShift )
% Local grid : 
% [ augBoxes, augImIds ] = bboxes_augmentation( gtBoxes, gtImIds, ...
%     [numGirdY numGridX numGridScale numGridAspect], useMirror, maxCenterShift, maxScaleShift, maxAspectShift )
% Local pyramid : 
% [ augBoxes, augImIds ] = bboxes_augmentation( gtBoxes, gtImIds, ...
%     [numGirdXY numGridScale numGridAspect], useMirror, maxCenterShift, maxScaleShift, maxAspectShift, minYXStride )

if iscell( gtBoxes )
    augBoxes = cell(size(gtBoxes));
    augIds = cell(size(gtIds));
    for k = 1:numel(gtBoxes)
        [augBoxes{k}, augIds{k}] = bboxes_augmentation( gtBoxes{k}, gtIds{k}, augFactor, useMirror, varargin{:} );
    end
    return;
end

%DEFAULT_SETTING.useMirror    = 1;
%DEFAULT_SETTING.maxJittering = 0.06;
%DEFAULT_SETTING.maxCenterShift = 0.5;
%DEFAULT_SETTING.maxScaleShift  = 2;

%DEFAULT_SETTING.gridSize       = [9 9 5 5];
%DEFAULT_SETTING.maxAspectShift = sqrt(2);

%DEFAULT_SETTING.pyramidSize    = [15 5 5];
%DEFAULT_SETTING.minYXStride    = 2;

switch length(varargin)
    case 1
        jitteringType = 'corners';
        maxJittering  = varargin{1};
    case 2
        jitteringType  = 'parameterized';
        maxCenterShift = varargin{1};
        maxScaleShift  = varargin{2};
    case 3
        jitteringType = 'grid';
        
        %DEFAULT_SETTING.maxCenterShift = 0.4;
        %DEFAULT_SETTING.maxScaleShift  = sqrt(2);
        
        if ~isempty(augFactor) && length(augFactor) ~= 4
            error( 'augFactor should be the grid size' );
        end
        
        gridSize = augFactor;
        maxCenterShift = varargin{1};
        maxScaleShift  = varargin{2};
        maxAspectShift = varargin{3};
    case 4
        
        jitteringType = 'pyramid';
        
        %DEFAULT_SETTING.maxCenterShift = 0.4;
        %DEFAULT_SETTING.maxScaleShift  = sqrt(2);
        
        if ~isempty(augFactor) && length(augFactor) ~= 3
            error( 'augFactor should be the pyramid size' );
        end
        
        pyramidSize = augFactor;
        maxCenterShift = varargin{1};
        maxScaleShift  = varargin{2};
        maxAspectShift = varargin{3};
        minYXStride    = varargin{4};
    otherwise
        error( 'Unrecognized parameters' );
end
    
% unstruct %DEFAULT_SETTING isempty always

if ismember( jitteringType, {'corners','parameterized'} )

    if augFactor<2
        augBoxes = gtBoxes;
        if (size(augBoxes,2)<5)
            augBoxes(:,5) = 0;
        end
        augIds = gtIds;
        return;
    end

    gtBoxes = gtBoxes(:,1:4);

    N = size(gtBoxes,1);

    augBoxes = repmat( gtBoxes, [1 1 augFactor] );
    augBoxes(:,1:4,1) = gtBoxes;

    switch jitteringType
        case 'corners'
            H = gtBoxes(:,3) - gtBoxes(:,1) + 1; % do we need to plus 1?
            W = gtBoxes(:,4) - gtBoxes(:,2) + 1;
            augBoxes(:,1:4,2:end) = ...
                bsxfun( @plus, gtBoxes, ...
                bsxfun( @times, [H,W,H,W], ...
                (rand( N, 4, augFactor-1 )*2-1)*maxJittering ) );    % jittering
        case 'parameterized'
            H = gtBoxes(:,3) - gtBoxes(:,1) + 1; % do we need to plus 1?
            W = gtBoxes(:,4) - gtBoxes(:,2) + 1;
            Y = mean( gtBoxes(:,[1 3]), 2);
            X = mean( gtBoxes(:,[2 4]), 2);
            randHW = exp( bsxfun( @plus, log([H, W]), ...
                (rand( N, 2, augFactor-1 )*2-1)* log(maxScaleShift) ) );
            randYX = bsxfun( @plus, [Y,X], ...
                bsxfun( @times, [H,W], ...
                (rand( N, 2, augFactor-1 )*2-1) * maxCenterShift ) );
            augBoxes(:,1:4,2:end) = round( [randYX-randHW/2, randYX+randHW/2] );
        otherwise 
            error( 'Unrecognized jittering type.' );
    end

    if useMirror
        augBoxes(:,5,3:end) = double(rand( N, 1, augFactor-2 )>0.5);    % random flips
        augBoxes(:,1:4,2) = gtBoxes;
        augBoxes(:,5,2)   = double(~augBoxes(:,5,2)); % mirror of the original bbox
    end


    augBoxes = reshape( permute( augBoxes, [2 1 3] ), ...
            [size(augBoxes,2),N*augFactor] ).';
    augIds = repmat( reshape(gtIds,numel(gtIds),1), augFactor, 1 );
    
elseif ismember( jitteringType, {'grid', 'pyramid'} )
        
    numBoxes = size( gtBoxes, 1 );
    
    [gY,gX,gH,gW] = bbox_ltrb2param( gtBoxes, 'yxhw' );
    [ ~, ~,gS,gA] = bbox_ltrb2param( [gY,gX,gH,gW], 'yxhw==>yxsal' );
    
    Yr = [gY,gY] + [-gH,gH].*maxCenterShift;
    Xr = [gX,gX] + [-gW,gW].*maxCenterShift;
    Sr = bsxfun( @plus, [gS,gS], [-1 1]*log(maxScaleShift ) );
    Ar = bsxfun( @plus, [gA,gA], [-1 1]*log(maxAspectShift) );

    augBoxes = cell( numBoxes, 1 );
    parfor j = 1:numBoxes
        augParams_j = {};
        switch jitteringType
            case 'grid'
                
                augParams_j = cell(4,1);
                R = [Yr(j,:);Xr(j,:);Sr(j,:);Ar(j,:)];
                G = arrayfun( @even_sampling_cell, ...
                    R(:,1), R(:,2), vec(gridSize) );
                [augParams_j{:}] = ndgrid( G{:} );
                augParams_j = cellfun( @(a) {vec(a)}, augParams_j );
                augParams_j = cat( 2, augParams_j{:} );
                
            case 'pyramid'
                
                Ryx = [Yr(j,:);Xr(j,:)];
                
                maxYXSize = ceil( (Ryx(:,2) - Ryx(:,1)).' / minYXStride );
                
                Ras = [Sr(j,:);Ar(j,:)];
                Gas = arrayfun( @even_sampling_cell, ...
                    Ras(:,1), Ras(:,2), vec(pyramidSize(2:3)) );
                [augS, augA] = ndgrid( Gas{:} );
                
                augParams_j = cell(4,pyramidSize(2),pyramidSize(3));
                for aspect_idx = 1:pyramidSize(3)
                    for scale_idx = 1:pyramidSize(2)
                        sc = augS(scale_idx,aspect_idx);
                        as = augA(scale_idx,aspect_idx);
                        
                        if any( (sc + as * [0.5,-0.5]) < log( minYXStride*2 ) )
                            continue;
                        end
                        
                        gridYXSize = round( (pyramidSize(1)-1) * ...
                            exp(gS(j)-sc) * exp(as * [0.5,-0.5]) )+1;
                        gridYXSize = min( gridYXSize, maxYXSize );
                        
                        Gyx = arrayfun( @even_sampling_cell, ...
                            Ryx(:,1), Ryx(:,2), vec(gridYXSize) );

                        [ augParams_j{:,scale_idx,aspect_idx} ] = ...
                            ndgrid( Gyx{:}, sc, as );
                    end
                end
                
                augParams_j = cellfun( @(a) {vec(a).'}, augParams_j );
                augParams_j = cat1dim( augParams_j(:,:), 2 );
                augParams_j = cellfun( @(a) {vec(a)}, augParams_j );
                augParams_j = cat( 2, augParams_j{:} );
            otherwise
                error( 'Internal error: unrecognized jitteringType' );
        end
        
        augBoxes_j = bbox_param2ltrb( augParams_j , 'yxsal' );
        augBoxes_j = [ augBoxes_j; gtBoxes(j,:) ];
        augBoxes_j = unique( round( augBoxes_j ), 'rows' );
        if size(gtBoxes,2)>=5
            augBoxes_j(:,5) = gtBoxes(j,5);
        end
        augBoxes{j} = augBoxes_j;
    end
    augIds = arrayfun( @(a,i) {repmat( i, size(a{1},1), 1 )}, ...
        augBoxes, vec(gtIds) );
    augIds = cat(1, augIds{:});
    augBoxes = cat(1, augBoxes{:});
    
    if useMirror
        augIds = [augIds;augIds];
        augBoxes = [augBoxes;augBoxes];
        if size(augBoxes,2)>=5
            augBoxes(end/2+1:end,5) = ~augBoxes(:,end/2+1:end);
        else
            augBoxes(end/2+1:end,5) = 1;
        end
    end
    
else
    error( 'Internal error : unrecognized jitteringType' );
end

end

function s = even_sampling( a, b, n )

if n == 1
    s = (a+b)/2;
else
    s = a:(b-a)/(n-1):b;
    % s = s + (b-s(end))/2;
end

end


function s = even_sampling_cell( st, en, n )

s = { even_sampling( st, en, n ) };

end

