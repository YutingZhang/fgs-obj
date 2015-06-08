clf

[THIS_FOLDER,~,~] = fileparts( mfilename('fullpath') );
addpath(THIS_FOLDER);

if exist('ACC_PATH','var')
    [ITER,ACC]=load_snapshot_accuracy(ACC_PATH);
else
    [ITER,ACC]=load_snapshot_accuracy;
end
validIdxB = ~isnan(ACC) & ~isinf(ITER);
plot(ITER(validIdxB),ACC(validIdxB),'.-b');
set(gca,'xgrid','on','ygrid','on');
xlabel('Iteration');
ylabel('Validation accuracy');

if any(isinf(ITER))
    refIdx = find(isinf(ITER),1);
    xlim_ = get(gca, 'xlim');
    line( [xlim_(1),xlim_(2)], [1,1]*ACC(refIdx), [0 0], 'color', 'red' );
    text( 0, ACC(refIdx), ' Reference model', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'bottom' );
end
