function [svm_model, cachePos, cacheNeg] = svm_str_loss_grad_multiclass...
    ( cachePos, idsBagPos, numBags, factorFeatScaling, ...
    funcBag, funcBagFilter, posOverlapThresh, negOverlapThresh, ...
    numMaxAddedNeg, numMaxAddedAmb, numEpoch, SVM_C, SVM_bias, SVM_PosW, ...
    SVM_ConstType, SVM_HingeLossType, LossCurveFunc)

% Remarks:
% cachePos should

fprintf( 1, '****** Hard SVM-Localization Train\n' );
numvis = size(cachePos{1}, 1);

numCategory = length( cachePos );

svm_struct_args_cmd = '-c %.10g -o 2 -v 1 -e 1e-3';
svm_struct_args = sprintf( svm_struct_args_cmd, SVM_C );
fprintf( 1, 'SVMstruct args : %s\n', svm_struct_args );


evict_neg_thresh = -.2;
evict_amb_thresh = -.1;
evict_amb_num_limit  = 10; %inf; %5;


hard_thresh  = -.0001;
hard_amb_num_limit = inf; %30;


% load all bboxes in each bag
% classify bboxes into neg and ambiguous pos (which need structure objective)

% set up svm struct param
svm_struct_param = struct();
svm_struct_param.loss_curve_func = LossCurveFunc; % linear
svm_struct_param.pos_overlap_thresh = posOverlapThresh;
svm_struct_param.neg_overlap_thresh = negOverlapThresh;
svm_struct_param.pos_weight   = SVM_PosW;
svm_struct_param.lossFn       = @SVMLoc_LossFunc;
svm_struct_param.constraintFn = @SVMLoc_ConstraintFunc;
svm_struct_param.featureFn    = @SVMLoc_FeatureFunc;
svm_struct_param.dimension    = size(cachePos{1}, 1) + 1; % add a bias term

% convert cachePos to bag-wise and struct array style (which should be further transformed to cell)
cachePos    = vec( cachePos );
cachePos    = cellfun( @(a) {[a ; ones(1,size(a,2))*SVM_bias]}, cachePos );

cachePos0 = cachePos;
cachePos  = cell(numCategory, 1);
labelPos  = cell(numCategory, 1);
for c = 1:numCategory,
    cachePos{c} = cell(numBags,1);
    labelPos{c} = cell(numBags,1);
    for k = 1:numBags
        chosenIdxB_ck = ( idsBagPos{c} == k );
        cachePos0_ck  = cachePos0{c}(:,chosenIdxB_ck);
        cachePos{c}{k}   = struct( 'data', ...
            mat2cell( cachePos0_ck, size(cachePos0_ck,1), ones(1,size(cachePos0_ck,2)) ), ...
            'box_ids', nan, 'overlap', 1 ); % we don't need to save the gt_box_ids
        labelPos{c}{k} = repmat( struct('label',1,'id',1,'overlap',1), ...
            1, size(cachePos0_ck,2) );
    end
end
clear cachePos0 bagIdsPos

% initialize
cacheNeg = repmat({repmat({struct('data',{},'box_ids',{},'overlap',{})},numBags,1)},numCategory,1);
labelNeg = repmat({repmat({struct('label',{},'id',{},'overlap',{})},numBags,1)},numCategory,1);

num_added_neg   = zeros(numCategory,1);
num_added_amb   = zeros(numCategory,1);
max_added_neg   = @(epoch_id) min(epoch_id*100000, 300000);
cur_svm_model = cell( numCategory, 1 );


% setting for gradient-based optimization
% addpath ../classifier/svm_structured_loss/;
% addpath(genpath('~/libdeepnets2/common/utils/minFunc_2012/'));

% structured loss function
lfunc = @(x) str_loss(x, negOverlapThresh, posOverlapThresh);

for c = 1:numCategory,
    cur_svm_model{c}.w = [0.01*randn(numvis, 1) ; 0];
end

options = struct;
options.maxIter = 20;
options.maxFunEvals = 30;
options.method = 'lbfgs';
options.display = 'off';

C1 = SVM_C*SVM_PosW;    % C for positive images
C2 = SVM_C;             % C for negative samples


for epoch_id = 1:numEpoch
    for k = 1:numBags
        
        fprintf( 1, 'Epoch %d / %d - %d / %d : ', epoch_id, numEpoch, k, numBags );
        
        tic
        
        last_round = (epoch_id==numEpoch) && (k==numBags);
        
        cur_sam_bag = funcBag(k);
        cur_sam_bag(end+1,:) = SVM_bias;
        
        for c = 1:numCategory,
            
            % **** get box info
            [valid_sam_idx, gt_ids, best_overlaps] = funcBagFilter(c, k);
            % Remark: gt_ids, best_overlaps is for all the samples
            % inlcuding the invalid ones *******
            
            % sam_anchor_ids == 0 for neg, otherwise for amb (ambiguous) pos
            
            % **** filter out existing samples
            existing_pos_box_ids = [cachePos{c}{k}.box_ids];
            existing_neg_box_ids = [cacheNeg{c}{k}.box_ids];
            existing_box_ids = [existing_pos_box_ids,existing_neg_box_ids];
            nonexisting_sam_idxb1 = ~ismember(valid_sam_idx,existing_box_ids); % nan is automatically ignored
            valid_sam_idx  = valid_sam_idx(nonexisting_sam_idxb1);
            gt_ids2        = gt_ids( nonexisting_sam_idxb1 );
            best_overlaps2 = best_overlaps( nonexisting_sam_idxb1 );
            
            % **** group samples to neg and ambiguous pos
            %  - neg (idx only)
            new_neg_idxb1 = (best_overlaps2<negOverlapThresh);
            new_neg_idx   = valid_sam_idx( new_neg_idxb1 );
            %  - amb
            new_amb_idx    = valid_sam_idx(~new_neg_idxb1);
            new_amb_gt_idx = gt_ids2(~new_neg_idxb1);
            
            % **** filter out easy samples
            if ~isempty(cur_svm_model{c})
                % not first round
                % find hard neg
                z_neg = vec(cur_svm_model{c}.w)' * cur_sam_bag(:,new_neg_idx);
                new_neg_idx = new_neg_idx(z_neg>-1+hard_thresh);
                % fine hard ambiguous pos
                tmp_new_amb_idx = [];
                tmp_new_amb_gt_idx = [];
                for j = unique(vec(new_amb_gt_idx)')
                    cur_new_amb_idxb1  = (new_amb_gt_idx == j);
                    cur_new_amb_idx    = new_amb_idx(cur_new_amb_idxb1);
                    cur_new_amb_gt_idx = new_amb_gt_idx(cur_new_amb_idxb1);
                    tmp_Y = struct( 'label', 1, 'id', 1, 'overlap', 1 );
                    tmp_X = struct( ...
                        'data',{ [cachePos{c}{k}(j).data(:,1),cur_sam_bag(:,cur_new_amb_idx)] }, ...
                        'overlap', [1, best_overlaps(cur_new_amb_idx)] );
                    % ^^ here we don't need box_ids
                    cur_hinge_scores = SVMLoc_HingeFunc( svm_struct_param, cur_svm_model{c}, tmp_X, tmp_Y );
                    cur_hinge_scores = cur_hinge_scores(2:end-1); % remove gt scores
                    
                    cur_hard_thresh = min( hard_thresh, ...
                        the_kth_min( cur_hinge_scores, hard_amb_num_limit ) );
                    
                    cur_hard_idxb = ( cur_hinge_scores >= cur_hard_thresh );
                    tmp_new_amb_idx    = [tmp_new_amb_idx, cur_new_amb_idx( cur_hard_idxb )];
                    tmp_new_amb_gt_idx = [tmp_new_amb_gt_idx, cur_new_amb_gt_idx( cur_hard_idxb )];
                end
                new_amb_idx    = tmp_new_amb_idx;
                new_amb_gt_idx = tmp_new_amb_gt_idx;
            end
            
            % **** merge samples (neg, and each ambiguous pos)
            % neg
            new_negs = struct( 'data', mat2cell( cur_sam_bag(:,new_neg_idx), ...
                size(cur_sam_bag,1), ones(1,length(new_neg_idx)) ), ...
                'box_ids', num2cell(new_neg_idx), ...
                'overlap', num2cell(best_overlaps(new_neg_idx)) );
            new_neg_labels = struct( 'label', -1, ...
                'overlap', num2cell(best_overlaps(new_neg_idx)), ...
                'id', 1 );
            cacheNeg{c}{k} = [cacheNeg{c}{k},new_negs];
            labelNeg{c}{k} = [labelNeg{c}{k},new_neg_labels];
            clear new_negs new_neg_labels
            % ambiguous pos
            for j = unique(vec(new_amb_gt_idx)')
                cur_new_amb_idxb1 = (new_amb_gt_idx == j);
                cur_new_amb_idx   = new_amb_idx(cur_new_amb_idxb1);
                tmp_X = struct( ...
                    'data',{ cur_sam_bag(:,cur_new_amb_idx) }, ...
                    'overlap', { best_overlaps(cur_new_amb_idx) }, ...
                    'box_ids', { cur_new_amb_idx } );
                cachePos{c}{k}(j).data    = [ cachePos{c}{k}(j).data,    tmp_X.data    ];
                cachePos{c}{k}(j).overlap = [ cachePos{c}{k}(j).overlap, tmp_X.overlap ];
                cachePos{c}{k}(j).box_ids = [ cachePos{c}{k}(j).box_ids, tmp_X.box_ids ];
                % note that no need to update the pos labels
            end
            
            % **** update added num
            num_added_neg(c) = num_added_neg(c) + length( new_neg_idx );
            num_added_amb(c) = num_added_amb(c) + length( new_amb_idx );
        end
        
        toc
        
        need_update = (num_added_neg>numMaxAddedNeg | ...
            num_added_amb>numMaxAddedAmb | last_round );
        
        % **** update models if needed
        if any(need_update),
            for c = vec(find(need_update)).'
                
                tS = tic;
                fprintf( 1, '===================== Update classifier %d\n', c);
                
                % set up training samples
                params = svm_struct_param;
                
                switch SVM_ConstType,
                    case 'maxsum',
                        params.patterns = num2cell( cat(2,cachePos{c}{:},cacheNeg{c}{:}) );
                        params.labels   = num2cell( cat(2,labelPos{c}{:},labelNeg{c}{:}) );
                        numPosSamples = sum( cellfun( @length, labelPos{c} ) );
                    case 'maxmax',
                        active_neg_image_idxb = ~cellfun( @isempty, cacheNeg{c} );
                        cNeg = cellfun( @(a) {struct('data',{[a.data]},...
                            'box_ids',{[a.box_ids]},'overlap',{[a.overlap]})}, cacheNeg{c}(active_neg_image_idxb) );
                        lNeg = repmat( {struct('label',-1,'overlap',0,'id',0)}, sum(active_neg_image_idxb), 1 );
                        cPos = num2cell( cat(2,cachePos{c}{:}) );
                        lPos = num2cell( cat(2,labelPos{c}{:}) );
                        if length(cNeg)<length(cPos)
                            cPos = cPos(1:length(cNeg));
                            lPos = lPos(1:length(cNeg));
                            fprintf( 'WARNING: only %d pos samples are used\n', length(cPos) );
                        end
                        params.patterns = [ cPos, cNeg.' ];
                        params.labels   = [ lPos, lNeg.' ];
                        numPosSamples = sum( cellfun( @length, lPos ) );
                        clear lPos cPos
                    case 'sumsum',
                        active_neg_image_idxb = ~cellfun( @isempty, cacheNeg{c} );
                        cNeg = cellfun( @(a) {struct('data',{[a.data]},...
                            'box_ids',{[a.box_ids]},'overlap',{[a.overlap]})}, cacheNeg{c}(active_neg_image_idxb) );
                        lNeg = repmat( {struct('label',-1,'overlap',0,'id',0)}, sum(active_neg_image_idxb), 1 );
                        cPos = num2cell( cat(2,cachePos{c}{:}) );
                        lPos = num2cell( cat(2,labelPos{c}{:}) );
                        
                        params.patterns = [ cPos, cNeg.' ];
                        params.labels   = [ lPos, lNeg.' ];
                        numPosSamples = sum( cellfun( @length, lPos ) );
                        clear lPos cPos
                    otherwise
                        error( 'Unknown Constraint Type : %s', SVM_ConstType);
                end
                
                
                % count the sample num
                numNegSamples = sum( cellfun( @length, labelNeg{c} ) );
                numAmbSamples = sum( cellfun( @(a) length(a.box_ids), ...
                    params.patterns(1:numPosSamples) ) ) - numPosSamples;
                fprintf( 1, 'Pos: %d , Amb: %d , Neg: %d\n', ...
                    numPosSamples, numAmbSamples, numNegSamples );
                
                % train classifier
                npos = 0;
                nneg = 0;
                for i = 1:length(params.patterns),
                    if params.labels{i}.label == 1,
                        npos = npos + 1;
                    elseif params.labels{i}.label == -1,
                        nneg = nneg + 1;
                    end
                end
                
                % reshape data
                xgt = zeros(numvis, npos);
                xph = cell(npos, 1);
                stloss = cell(npos, 1);
                xnh = cell(nneg, 1);
                
                npos = 0;
                nneg = 0;
                for i = 1:length(params.patterns),
                    if params.labels{i}.label == 1,
                        npos = npos + 1;
                        xgt(:, npos) = params.patterns{i}.data(1:end-1, 1);
                        xph{npos} = double(params.patterns{i}.data(1:end-1, 2:end));
                        stloss{npos} = double(lfunc(params.patterns{i}.overlap(2:end)));
                    elseif params.labels{i}.label == -1,
                        nneg = nneg + 1;
                        xnh{nneg} = double(params.patterns{i}.data(1:end-1, :));
                    end
                end
                xgt = double(xgt);
                
                % L1-loss
                fprintf('class = %d, positive: %d, negative: %d\n', c, npos, nneg);
                w = cur_svm_model{c}.w;
                if strcmp(SVM_HingeLossType, 'L1')
                    if strcmp(SVM_ConstType, 'maxmax') || strcmp(SVM_ConstType, 'maxsum'),
                        compute_loss = @(x) compute_str_loss_max_L1(x, xgt, xph, stloss, xnh, C1, C2, SVM_bias);
                    elseif strcmp(SVM_ConstType, 'sumsum'),
                        compute_loss = @(x) compute_str_loss_sum_L1(x, xgt, xph, stloss, xnh, C1, C2, SVM_bias);
                    end
               else
                   error('Unknown SVM_HingleLossType');
               end
                
                w = minFunc(@(p) compute_loss(p), w, options);
                
                cur_svm_model{c}.w = w;
                clear xgt xph stloss xnh;
                
                
                % evict easy samples
                %  - neg
                tmp_neg_cnt = cellfun( @(a) size(a,2), cacheNeg{c} );
                tmp_negs    = cat(2,cacheNeg{c}{:});
                z_neg = vec(cur_svm_model{c}.w).' * cat(2,tmp_negs.data);
                
                % dynamic thresholding
                z_neg_sorted = sort(z_neg, 'descend');
                z_neg_thresh = z_neg_sorted(min(max_added_neg(epoch_id), length(z_neg_sorted)));
                
                tmp_evict_neg_idxb = (z_neg < max(-1+evict_neg_thresh, z_neg_thresh));
                evict_neg_idxb = mat2cell( tmp_evict_neg_idxb, 1, tmp_neg_cnt );
                for kk = 1:numBags
                    cacheNeg{c}{kk}(evict_neg_idxb{kk}) = [];
                    labelNeg{c}{kk}(evict_neg_idxb{kk}) = [];
                end
                clear tmp_negs
                numEvictNeg = sum(tmp_evict_neg_idxb);
                
                %  - amb pos
                numEvictAmb = 0;
                for kk = 1:numBags
                    for jj = 1:length( cachePos{c}{kk} )
                        if length(cachePos{c}{kk}(jj).box_ids)>1
                            cur_hinge_scores   = SVMLoc_HingeFunc( svm_struct_param, cur_svm_model{c}, ...
                                cachePos{c}{kk}(jj), labelPos{c}{kk}(jj) );
                            
                            cur_evict_amb_thresh = max( evict_amb_thresh, ...
                                the_kth_max( cur_hinge_scores, evict_amb_num_limit ) );
                            
                            cur_evict_amb_idxb = ( cur_hinge_scores(1:end-1) < cur_evict_amb_thresh );
                            
                            cur_evict_amb_idxb(1) = false;
                            cachePos{c}{kk}(jj).data(:,cur_evict_amb_idxb)  = [];
                            cachePos{c}{kk}(jj).overlap(cur_evict_amb_idxb) = [];
                            cachePos{c}{kk}(jj).box_ids(cur_evict_amb_idxb) = [];
                            numEvictAmb = numEvictAmb + sum(cur_evict_amb_idxb);
                        end
                    end
                end
                
                % clear added num
                num_added_neg(c) = 0;
                num_added_amb(c) = 0;
                
                fprintf( 1, '== Evicted Neg: %d , Remaining Neg: %d \n', numEvictNeg, numNegSamples-numEvictNeg );
                fprintf( 1, '== Evicted Amb: %d , Remaining Amb: %d \n', numEvictAmb, numAmbSamples-numEvictAmb );
                fprintf( 1, 'negative threshold = %g , time = %g \n\n', max(-1+evict_neg_thresh, z_neg_thresh), toc(tS));
            end
        end
    end
end

cur_svm_model_struct = cell2mat( cur_svm_model );
for c = 1:numel(  cur_svm_model_struct )
    cur_svm_model_struct(c).w = vec( cur_svm_model_struct(c).w ).';
end

svm_model = struct( 'scale', factorFeatScaling, ...
    'w', {cat(1,cur_svm_model_struct.w)} );
svm_model.bias = svm_model.w(:,end)*SVM_bias;
svm_model.w    = svm_model.w(:,1:end-1);

% remove data and keep box_ids and overlap only
for c = 1:length(cachePos),
    for kk = 1:length(cachePos{c}),
        if isfield(cachePos{c}{kk}, 'data'),
            cachePos{c}{kk} = rmfield(cachePos{c}{kk}, 'data');
        end
    end
end

for c = 1:length(cacheNeg),
    for kk = 1:length(cacheNeg{c}),
        if isfield(cacheNeg{c}{kk}, 'data'),
            cacheNeg{c}{kk} = rmfield(cacheNeg{c}{kk}, 'data');
        end
    end
end

end

function km = the_kth_min(a,k)

if k>numel(a)
    km = inf;
elseif k==numel(a)
    km = min(a);
else
    b  = sort(a,'ascend');
    km = b(k);
end

end

function km = the_kth_max(a,k)

if k>numel(a)
    km = -inf;
elseif k==numel(a)
    km = max(a);
else
    b  = sort(a,'descend');
    km = b(k);
end

end
