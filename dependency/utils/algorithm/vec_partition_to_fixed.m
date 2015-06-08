function [batch_splitters, batch_sizes] = vec_partition_to_fixed( v, maximum_batch_size )

c = [0; cumsum( v )];
st = 1; imgCollectionNum = 0;
batch_splitters = zeros(length(v),2);
while st<=length(v)
    en = find( c <= c(st)+maximum_batch_size, 1, 'last' ) - 1;
    % en = max( en, st+1 );   % must move forward
    imgCollectionNum = imgCollectionNum + 1;
    batch_splitters(imgCollectionNum,:) = [st en];
    st = en + 1;
end
batch_splitters  = batch_splitters(1:imgCollectionNum,:);
batch_sizes      = batch_splitters(:,2)-batch_splitters(:,1)+1;

end
