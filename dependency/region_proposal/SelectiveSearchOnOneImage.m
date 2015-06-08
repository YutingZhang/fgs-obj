function bboxes = SelectiveSearchOnOneImage( im, param )
% Usage: bboxes = SelectiveSearchOnOneImage( im, param )
% param: 'ijcv_fast', 'ijcv_quality', 'ijcv_fast', or a struct like
%     param.colorTypes = {'Hsv', 'Lab'};
%     param.simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill};
%     param.ks = [50 100];
%     param.sigma = 0.8;
%     param.minBoxWidth = 20;
% bboxes = [y1 x1 y2 x2]

%% parameters

sigma = []; % sigma from unstruct cannot override the sigma function (seems a matlab bug)

if ischar( param )
    switch param
        case 'ijcv_fast'

            colorTypes = {'Hsv', 'Lab'};
            simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill};
            ks = [50 100];

        otherwise
            error('Unrecognized parameter');
    end
end

%   ijcv_fast settings as the default
sigma = 0.8;
minBoxWidth = 20;

%% processing
Nks = length(ks);
Nct = length(colorTypes);

boxesT = cell(1,Nks*Nct);
priorityT = cell(1,Nks*Nct);
% tic;
% parfor idx=1:length(boxesT)
for idx=1:length(boxesT)
    ks1 = ks; colorTypes1 = colorTypes;
    [n,j] = ind2sub( [Nct,Nks], idx );
    k0 = ks1(j); % Segmentation threshold k
    minSize = k0; % We set minSize = k
    colorType = colorTypes1{n};
    [boxesT{idx} blobIndIm blobBoxes hierarchy priorityT{idx}] = ...
        Image2HierarchicalGrouping(im, sigma, k0, minSize, ...
        colorType, simFunctionHandles);
end
% toc;
bboxes = cat(1, boxesT{:}); % Concatenate bboxes from all hierarchies
priority = cat(1, priorityT{:}); % Concatenate priorities

% Do pseudo random sorting as in paper
priority = priority .* rand(size(priority));
[priority, sortIds] = sort(priority, 'ascend');
bboxes = bboxes(sortIds,:);

%
bboxes = FilterBoxesWidth(bboxes, minBoxWidth);
bboxes = BoxRemoveDuplicates(bboxes);

end
