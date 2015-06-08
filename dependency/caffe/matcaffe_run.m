function feat = matcaffe_run( patches, response_ids )
% feat = matcaffe_run(patches, response_ids)

global caffe_batch_size

% prepare input
patches = single( patches );

if ( size(patches,3)==3 )
    patches = patches( :,:,[3 2 1], : );
end
patches = permute( patches, [2 1 3 4] );

N = size( patches, 4 ); 
batch_num = ceil( N/caffe_batch_size );
last_batch_size = N - caffe_batch_size * (batch_num-1);
if N==1
    % suppress warning
    input_data = {patches};
else
    input_data = mat2cell( patches, size(patches,1), size(patches,2), size(patches,3), ...
        [ repmat( caffe_batch_size, 1, batch_num-1 ), last_batch_size ] );
end
input_data{end}(:,:,:,last_batch_size+1:caffe_batch_size) = NaN;    % fill the batch

% do forward pass to get scores

M = length(response_ids);
feat = cell([1,batch_num]);
for k = 1:batch_num
    for numTry = 1:2
        ft = caffe( 'forward', input_data(k) );
        feat{k} = caffe( 'response', response_ids );
        is_secured = true;
        for i = 1:M
            feat{k}{i} = reshape( feat{k}{i}, numel(feat{k}{i})/caffe_batch_size, caffe_batch_size );
            is_secured = is_secured & all( abs(feat{k}{i}(:))<1e5 ); % test whether something weird happens
        end
    end
end
for i = 1:M
    feat{end}{i} = feat{end}{i}(:,1:last_batch_size);
end

feat1 = cat(2, feat{:} );
feat = cell(M,1);
for i = 1:M
    feat{i} = cat(2, feat1{i,:});
end

end
