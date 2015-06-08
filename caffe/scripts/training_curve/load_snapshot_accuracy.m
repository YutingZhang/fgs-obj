function [ITER, ACC] = load_snapshot_accuracy(ACC_PATH)

if ~exist('ACC_PATH','var')
    ACC_PATH = '';
end

fl = dir( fullfile(ACC_PATH,'*.accuracy') );
fl = {fl.name};

ITER = zeros(length(fl),1);
ACC  = cell(length(fl),1);

for i = 1:length(fl)
    [ ~,fn,~ ] = fileparts( fl{i} );
    iter_start_pos= length(fn) - ...
        find( ismember(fn(end:-1:1),char((0:9)+'0')), 1, 'last' )+1;
    iter_ = str2double( fn( iter_start_pos:end ) );
    if (isempty(iter_))
        ITER(i) = inf;
    else
        ITER(i) = iter_;
    end
    fid = fopen( fullfile(ACC_PATH,fl{i}), 'r' );
    ACC{i}  = fscanf( fid, '%f' );
    fclose(fid);
end
ACC( cellfun(@isempty,ACC) ) = {nan};
ACC = cell2mat(ACC);
[ ITER, sortedIdx ] = sort(ITER);
ACC = ACC(sortedIdx);

end

