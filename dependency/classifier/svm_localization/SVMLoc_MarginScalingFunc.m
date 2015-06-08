function [scores, yhats] = SVMLoc_MarginScalingFunc(param, model, x, y)
% margin rescaling

% NOTE: y's label is always the groundtruth label of x

% batch compute loss

w = reshape( model.w, 1, numel(model.w) );

if y.label>0
    
    if y.overlap ~=1
        error( 'y.overlap (y is a pos groundtruth) must be 1' );
    end
    
    l = SVMLoc_OverlapLoss( param, x.overlap );

    l(end+1) = 1;   % for y = -1
    s = [w * x.data, 0];   % 1 is the groundtruth with overlap 1
    
    yhats = struct( ...
        'id', num2cell([1:size(x.data,2), 1]), ...
        'overlap', num2cell([x.overlap 1]), ...
        'label', num2cell([ones(1,size(x.data,2)),-1]) );
    
else %y<=0
    
    %{
    s = [ 0, w * x.data(:,1)];
    l = [ 0 1 ];
    yhats = struct( 'id', {1, 1}, 'overlap', {0, 0}, 'label', {-1, 1} );
    %}

    n = size(x.data,2);
    s = [ zeros(1,n), w * x.data];
    l = [ zeros(1,n), ones(1,n) ];
    yhats = struct( 'id', num2cell([1:n,1:n]), ...
        'overlap', num2cell(zeros(1,2*n)), ...
        'label',   num2cell([-ones(1,n) ones(1,n)]) );


end

% classification score

scores = l+s;

end
