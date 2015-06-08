function [newBs, newSs] = detSingleGP( I, bboxes, S, det_model, gp_thresh, ...
    CategIdOfInterest, displayProposal )
% DETSINGLEGP performs GP-based Fine-grained Search (FGS) on a single image with initial detection outputs
% 
% Usage:
%
%   [newBs, newSs] = detSingleGP( I, bboxes, S, det_model, thresh, CategIdOfInterest, displayProposal )
%
% Input:
%
%   I: an image matrix loaded by imread (e.g., I = imread('000220.jpg');) 
%
%   det_model: is the detection model loaded by detInit(...)
%
%   bboxes: can be M*4 maxtrix for intial bounding box coordinates, where 
%     M is the number of initial bounding boxes. Each row should be in the
%     form of [ymin, xmin, ymax, xmax].
%
%   gp_thresh: threshold for candidate region for the GP-based FGS.
%       Default value: -1
%
%   CategOfInterest4GP: can be a string cell array indicating which
%     categories FGS should be applied. It is only useful when gp_enable==1
%     By default FGS is applied to all the categories.
%     E.g. CategOfInterest4GP = {'aeroplane','cow'}
%
%   displayProposal: can be 0 (default) or 1 to indicate whether to show
%     step-by-step FGS proposals
%
% Output: Let N be the number of categories
%
%   newBs : N-d cell array. Bs{i} is the coordinates of the bounding boxes
%     proposed by FGS for the i-th category, which is a matrix with 4 columns.
%     Each row should be in the form of [ymin, xmin, ymax, xmax].
%
%   newSs : N-d cell array. Ss{i} is a vector of the scores of the bounding 
%     boxes proposed by FGS for the i-th category,
%
%

if ~exist('thresh','var') || isempty(gp_thresh)
    gp_thresh = -1;
end
if isscalar( gp_thresh )
    gp_thresh = repmat(gp_thresh, 1, length( det_model.categ_list ));
end

if ~exist('CategIdOfInterest','var') || isempty(CategIdOfInterest)
    CategIdOfInterest = 1:length(det_model.categ_list);
end
CategIdOfInterest = reshape(CategIdOfInterest,1,numel(CategIdOfInterest));

if ~exist('displayProposal','var') || isempty(displayProposal)
    displayProposal = 0;
end


newBs = cell(length(det_model.categ_list),1);
newSs = cell(length(det_model.categ_list),1);

%% Prepare models
classifier_model = cell(length(CategIdOfInterest),1);
GPmodel = cell(length(CategIdOfInterest),1);
for c = 1:length(CategIdOfInterest)
    classifier_model{c} = struct( ...
        'w',     {det_model.classifier.w(c,:)}, ...
        'bias',  {det_model.classifier.bias(c)}, ...
        'type',  {det_model.classifier.type} );
    GPmodel{c} = sgp_model_from_general( det_model.gp(c).hyp );
end
classifier_model = cell2mat(classifier_model);
GPmodel = cell2mat(GPmodel);

feat_func = @(varargin) ind( det_model.cnn.feat_func(varargin{:}), 1 );

%% set up solver

Solver_Timeout = 10;
minFunc_Method = 'lbfgs';
minFuncX_OPTS = struct();
minFuncX_OPTS.timeout = Solver_Timeout;
minFuncX_OPTS.Display = 'off';
minFuncX_OPTS.Method  = minFunc_Method;
min_func = @(func,x0) minFuncX( func,x0, minFuncX_OPTS);


%% GPSearch
bboxParamType = det_model.gp(1).BBoxParamType;

scoreThreshold         = gp_thresh;
maxGPIter              = 32;
maxGPGapNum            = 8;
maxLocalityNumPerImage = inf;

nmsThreshold       = 0.3;
localIoUThresholds = sort([0.3,0.5,0.7], 'ascend' );

baggingNum = length(localIoUThresholds);
categNum = length(CategIdOfInterest);

% set up FGS input

curScores = cell(categNum,1);
curBoxes  = bboxes;
[curBoxes, ia] = unique(curBoxes,'rows','last');
curBoxes  = curBoxes(ia,:);

for c = 1:categNum
    curScores{c} = vec(S(c,ia));
end

imHeight = size(I,1);
imWidth  = size(I,2);

% ============ FGS procedure START

curBParams = bbox_ltrb2param( curBoxes, bboxParamType );
curBoxes   = repmat( {curBoxes}, categNum, 1 );
curBParams = repmat( {curBParams}, categNum, 1 );

gapIterNum = zeros(categNum,1);
activeClassIdx = 1:categNum;

for iter=1:maxGPIter

    nextBParams = cell(categNum,1);
    nextOrigin  = cell(categNum,1);
    bestScores  = cell(categNum,1);

    for c = activeClassIdx

        goodIdxB = ( curScores{c}>=scoreThreshold(c) );
        goodIdx  = find( goodIdxB );
        furtherGoodIdx  = nms( [curBoxes{c}(goodIdxB,:), ...
            curScores{c}(goodIdxB)], nmsThreshold, 'iou' );
        anchorIdx = goodIdx(furtherGoodIdx);

        if isempty(anchorIdx)
            activeClassIdx = setdiff( activeClassIdx, c );
            continue;
        end

        anchorBoxes  = curBoxes{c}(anchorIdx,:);
        anchorScores = curScores{c}(anchorIdx);

        [~,further2BestIdx] = sort( anchorScores, 'ascend' );
        further2BestIdx = further2BestIdx( ...
            1:min( length(further2BestIdx), maxLocalityNumPerImage ) );
        anchorScores  = anchorScores(further2BestIdx);
        anchorBoxes   = anchorBoxes(further2BestIdx,:);
        anchorBParams = bbox_ltrb2param( anchorBoxes, bboxParamType );
        [~,~,anchorScales,~]  = bbox_ltrb2param( anchorBoxes, 'yxsal' );

        % local gp
        cur_IoU = PairedIoU( curBoxes{c}, anchorBoxes );

        nextBParams{c} = cell(size(anchorBoxes,1),baggingNum);
        nextOrigin{c}  = cell(size(anchorBoxes,1),baggingNum);
        bestScores{c}  = cell(size(anchorBoxes,1),baggingNum);

        for j=1:size(anchorBoxes, 1)
            for bag_id = 1:baggingNum
                localIdxB = (cur_IoU(:,j)>localIoUThresholds(bag_id));
                if sum(localIdxB)<3
                    break; % note that localIoUThresholds is in ascending order
                end

                PsiN1 = curBParams{c}(localIdxB,:).';
                fN    = curScores{c}(localIdxB);
                if bag_id == 1
                    fN_hat = max(fN);
                end
                
                latent_obj = @(z) sgp_negloglik( GPmodel(c), z, PsiN1, fN );
                z0 = anchorScales(j);
                try
                    z_hat = min_func( latent_obj, z0);
                catch
                    % warning( 'Optimization on z is failed' );
                    z_hat = anchorScales(j);
                end

                expnz = exp(-z_hat);

                PsiN = PsiN1;
                PsiN(GPmodel(c).idxbScaleEnabled,:)  = PsiN(GPmodel(c).idxbScaleEnabled,:)*expnz;
                KN = sgp_cov( GPmodel(c), 0, PsiN );

                search_obj = @(psiNp1) sgp_neg_acquisition_ei( GPmodel(c), ...
                    psiNp1, PsiN, fN, fN_hat, KN );

                psiNp1_0 = anchorBParams(j,:).';
                psiNp1_0(GPmodel(c).idxbScaleEnabled) = psiNp1_0(GPmodel(c).idxbScaleEnabled)*expnz;
                try
                    psiNp1_hat = min_func( search_obj, psiNp1_0 );
                catch
                    warning( 'Optimization on psiNp1_hat is failed' );
                    continue;
                end
                psiNp1_hat_1 = psiNp1_hat;
                psiNp1_hat_1(GPmodel(c).idxbScaleEnabled) = psiNp1_hat_1(GPmodel(c).idxbScaleEnabled) / expnz;

                if displayProposal
                    if isempty(I)
                        I = imread( TEST_DATA_LIST(k).im );
                    end
                    pbox = bbox_param2ltrb( psiNp1_hat_1.', bboxParamType );
                    show_bboxes(I,anchorBoxes(j,:),[],'green');
                    show_bboxes([],pbox,[],'yellow');
                    keyboard
                end

                bestScores{c}{j, bag_id} = fN_hat;
                nextBParams{c}{j,bag_id} = psiNp1_hat_1.';
                nextOrigin{c}{j,bag_id}  = [c,j,bag_id];
            end
        end

    end

    % put things together
    bestScores_noncell  = cell(categNum,1);
    nextBParams_noncell = cell(categNum,1);
    nextOrigin_noncell  = cell(categNum,1);
    for c = activeClassIdx
        bestScores_noncell{c}  = cat(1,bestScores{c}{:});
        nextBParams_noncell{c} = cat(1,nextBParams{c}{:});
        nextOrigin_noncell{c}  = cat(1,nextOrigin{c}{:});
    end
    bestScores_noncell  = cat(1,bestScores_noncell{:});
    nextBParams_noncell = cat(1,nextBParams_noncell{:});
    nextOrigin_noncell  = cat(1,nextOrigin_noncell{:});

    if isempty(bestScores_noncell)
        fprintf('x');
        break;
    end

    nextBoxes_noncell = round( bbox_param2ltrb( nextBParams_noncell, bboxParamType ) );

    % pruning apparent bad solutions
    prunedIdxB              = any( isnan(nextBoxes_noncell), 2 ) | any( abs(nextBoxes_noncell)>1e5, 2 );
    prunedIdxB(~prunedIdxB) = any( nextBoxes_noncell(~prunedIdxB,[3 4])<nextBoxes_noncell(~prunedIdxB,[1 2]),2);
    prunedIdxB(~prunedIdxB) = ...
        nextBoxes_noncell(~prunedIdxB,3)<1 | ...
        nextBoxes_noncell(~prunedIdxB,1)>imHeight | ...
        nextBoxes_noncell(~prunedIdxB,4)<1 | ...
        nextBoxes_noncell(~prunedIdxB,2)>imWidth;

    bestScores_noncell(prunedIdxB)    = [];
    nextOrigin_noncell(prunedIdxB,:)  = [];
    nextBoxes_noncell(prunedIdxB,:)   = [];

    if isempty(bestScores_noncell)
        fprintf('x');
        break;
    end

    % pruning duplicated bboxes
    [bestScores_noncell, sorted_idx] = sort(bestScores_noncell,'ascend');
    nextBoxes_noncell = nextBoxes_noncell(sorted_idx,:);
    nextOrigin_noncell = nextOrigin_noncell(sorted_idx,:);
    [nextBoxes_noncell, uq_idx] = unique([nextBoxes_noncell,nextOrigin_noncell(:,1)],'rows','first','legacy');
    nextBoxes_noncell  = nextBoxes_noncell(:,1:4);
    bestScores_noncell = bestScores_noncell(uq_idx); % use the lowest best score
    nextOrigin_noncell = nextOrigin_noncell(uq_idx,:);

    dupIdxB = false( size(bestScores_noncell) );
    for c = activeClassIdx
        thisIdxB          = (nextOrigin_noncell(:,1) == c);
        dupIdxB(thisIdxB) = ismember( nextBoxes_noncell(thisIdxB,:), curBoxes{c}, 'rows' );
    end
    bestScores_noncell(dupIdxB)    = [];
    nextOrigin_noncell(dupIdxB,:)  = [];
    nextBoxes_noncell(dupIdxB,:)   = [];

    if isempty(bestScores_noncell)
        fprintf('x');
        break;
    end

    nextBParams_noncell = bbox_ltrb2param( nextBoxes_noncell, bboxParamType );

    % extract features
    [uqBoxes, ~, ci] = unique(nextBoxes_noncell,'rows');
    
    uqF = features_from_bboxes( I, uqBoxes, ...
        det_model.cnn.canonical_patchsize, ...
        det_model.cnn.padding, feat_func, ...
        det_model.cnn.max_batch_num * det_model.cnn.batch_size  );
    uqF = cell2mat( uqF );
    nextF_noncell = uqF(:,ci);

    % compute scores
    gapIterNum = gapIterNum + 1;
    for c = activeClassIdx
        thisIdxB = (nextOrigin_noncell(:,1) == c);
        if any(thisIdxB)
            nextScores_c = ApplyClassifier( nextF_noncell(:,thisIdxB), classifier_model(c) ).';
            if any( nextScores_c>bestScores_noncell(thisIdxB) )
                gapIterNum(c) = 0;
            end
            curScores{c} = [curScores{c};nextScores_c];
            curBoxes{c}  = [curBoxes{c};nextBoxes_noncell(thisIdxB,:)];
            curBParams{c}= [curBParams{c};nextBParams_noncell(thisIdxB,:)];
        end
    end

    if ~all(gapIterNum)
        fprintf('*');
    else
        fprintf('.');
    end

    activeClassIdxB = false(1,categNum);
    activeClassIdxB(activeClassIdx) = true;
    activeClassIdxB(gapIterNum>=maxGPGapNum) = false;
    activeClassIdx = find(activeClassIdxB);

    if isempty(activeClassIdx), break; end

end

for c = 1:categNum
    c1 = CategIdOfInterest(c);
    newN_c = length( curScores{c} ) - size(S,2);
    newBs{c1} = curBoxes{c}(end-newN_c+1:end,:);
    newSs{c1} = curScores{c}(end-newN_c+1:end);
end

