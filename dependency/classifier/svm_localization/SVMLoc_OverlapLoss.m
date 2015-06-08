function delta = SVMLoc_OverlapLoss(param, ov)

essential_overlap = max( 0, min( 1, (ov - param.neg_overlap_thresh) ./ ...
    ( param.pos_overlap_thresh - param.neg_overlap_thresh ) ) );

non_overlap = 1 - essential_overlap;

delta = param.loss_curve_func(non_overlap);

end
