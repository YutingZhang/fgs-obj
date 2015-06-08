function scores = SVMLoc_HingeFunc(param, model, x, y)

% NOTE: y's label is always the groundtruth label of x

s_margin = SVMLoc_MarginScalingFunc(param, model, x, y);

w    = reshape( model.w, 1, numel(model.w) );
s_gt = (0.5 * y.label * w) * x.data(:,1);

scores = s_margin - s_gt;

end
