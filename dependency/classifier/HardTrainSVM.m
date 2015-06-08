function svm_model = HardTrainSVM( cachePos, numMaxAddedNeg, numEpoch, ...
    numNegBags, funcNegBag, funcNegBagFilter, ...
    factorFeatScaling, SVM_Backend, SVM_Type, SVM_C, SVM_bias, SVM_PosW, ...
    CategList, Epoch_Callback )

fprintf( 1, '****** Hard SVM Train\n' );

if ~exist( 'Epoch_Callback', 'var' ) || isempty(CategList)
    Epoch_Callback = @(cl,ep) dummy(cl,ep);
end

numCategory = length( cachePos );
if ~exist( 'CategList', 'var' ) || isempty(CategList)
    tdn = ceil(log10(numCategory));
    CategList = arrayfun( @(a) {sprintf(['Class %' int2str(tdn) 'd'],a)}, 1:numCategory );
end

switch SVM_Backend
    case 'liblinear';
        backend_opts_cmd = sprintf('-w1 %.5g -c %s -s %d -B %.5g', ...
            SVM_PosW, '%.10g', SVM_Type, SVM_bias);
    otherwise
        error( 'Unsupported SVM_Backend' );
end

if isscalar(SVM_C)
    backend_opts = sprintf( backend_opts_cmd, SVM_C );
    fprintf( 1, 'Backend options : %s\n', backend_opts );
    backend_opts = repmat( {backend_opts}, 1, numCategory );
else
    SVM_C = vec(SVM_C).';
    backend_opts = arrayfun( @(c) {sprintf( backend_opts_cmd, c )}, SVM_C );
    fprintf( 1, 'Backend options : \n' );
    mcnl = max( cellfun( @length, CategList ) );
    for c = 1:numCategory
        fprintf( 1, ['  - %' int2str(mcnl) 's : %s\n'], CategList{c}, backend_opts{c} );
    end
end


evict_thresh = -1.2;
hard_thresh  = -1.0001;

cachePos    = vec( cachePos );
cachePos    = cellfun( @(a) {a*factorFeatScaling}, cachePos );

cacheNeg    = cell(numCategory,1);
keyNeg      = cell(numCategory,1); % [bag_id; sample_id]

num_added   = zeros(numCategory,1);

cur_svm_model = cell( numCategory, 1 );

for epoch_id = 1:numEpoch
    for k = 1:numNegBags
        
        fprintf( 1, 'Epoch %d / %d - %d / %d : ', ...
            epoch_id, numEpoch, k, numNegBags );
        
        any_update = 0;
        
        tic
        
        last_round = (epoch_id==numEpoch) && (k==numNegBags);
        
        cur_neg_bag = funcNegBag(k) * factorFeatScaling;
        
        for c = 1:numCategory
            
            valid_neg_idx = funcNegBagFilter(c,k);
            
            if ~isempty(cur_svm_model{c})
                % not first round
                % find hard samples
                z_neg = bsxfun( @plus, cur_svm_model{c}.w*cur_neg_bag(:,valid_neg_idx), ...
                    cur_svm_model{c}.b );
                hard_idxb = (z_neg>hard_thresh);
                valid_neg_idx = valid_neg_idx( hard_idxb );
            end
            
            if isempty( keyNeg{c} ) 
                cacheNeg{c} = cur_neg_bag(:,valid_neg_idx);
                keyNeg{c}   = [repmat(k,1,numel(valid_neg_idx));vec(valid_neg_idx).'];
            else
                
                is_duplicated = ismember( valid_neg_idx, ...
                    keyNeg{c}( 2, keyNeg{c}(1,:) == k ) );
                valid_neg_idx = valid_neg_idx(~is_duplicated);
                
                cacheNeg{c} = [ cacheNeg{c}, cur_neg_bag(:,valid_neg_idx) ];
                keyNeg{c}   = [ keyNeg{c} , ...
                    [repmat(k,1,numel(valid_neg_idx));vec(valid_neg_idx).'] ];
                
            end
            num_added(c) = num_added(c) + numel(valid_neg_idx);
            
            need_update = (num_added(c)>numMaxAddedNeg || last_round );
            if need_update
                
                if ~any_update
                    any_update = 1;
                    fprintf( 1, '\n' );
                end
                
                fprintf( 1, '===================== Update classifier for Class %d - %s \n', c, CategList{c} );
                % train classifier
                num_pos = size( cachePos{c}, 2 );
                num_neg = size( cacheNeg{c}, 2 );
                fprintf(1,'== Pos: %d , Neg: %d ( Added: %d )\n',...
                    num_pos,num_neg,num_added(c));
                fprintf('liblinear opts: %s\n', backend_opts{c});
                X = [sparse(double(cachePos{c})), sparse(double(cacheNeg{c}))];
                y = cat(1, ones(num_pos,1), -ones(num_neg,1));
                llm = liblinear_train(y, X, backend_opts{c}, 'col');
                
                cur_svm_model{c} = struct( ...
                    'w', vec(llm.w(1:end-1)).', ...
                    'b', llm.w(end)*SVM_bias );
                % compute loss
                %  -- leave it blank 
                
                % evict easy example
                
                if ~isempty( cacheNeg{c} )
                    z_neg = bsxfun( @plus, cur_svm_model{c}.w * cacheNeg{c}, cur_svm_model{c}.b );
                    evict_idxb = (z_neg<evict_thresh);

                    cacheNeg{c} = cacheNeg{c}( :, ~evict_idxb );
                    keyNeg{c}   = keyNeg{c}( :, ~evict_idxb );
                    numEvicted = sum(evict_idxb);
                else
                    numEvicted = 0;
                end
                
                fprintf( 1, '== Evicted: %d , Remaining Neg: %d\n', numEvicted, size(keyNeg{c},2) );
                
                num_added(c) = 0;
                
            end
            
        end
        
        if any_update
            fprintf( 1, '*** ' );
        end
        toc
        
    end
    
    cur_svm_model_struct = cell2mat( cur_svm_model );
    svm_model = struct( 'scale', factorFeatScaling, ...
        'w', {cat(1,cur_svm_model_struct.w)}, 'bias', {cat(1,cur_svm_model_struct.b)} );
    
    Epoch_Callback( svm_model, epoch_id );
    
end



end
