function F = features_from_bboxes( I, boxes, canonical_patchsize, padding, feat_func, collection_size  )

if collection_size<inf

    boxN = size(boxes,1);
    boxBatchN = ceil( boxN/collection_size );
    F = cell( 1, boxBatchN );
    for b = 1:boxBatchN     % this loop is for reducing memory usages
        st1 = (b-1)*collection_size+1;
        en1 = min( b*collection_size, boxN );
        patches = extract_patches_from_image( single(I), boxes(st1:en1,:), ...
            canonical_patchsize, padding );
        F{b} = feat_func( patches );
    end
    F = [ F{:} ];
    F = cat1dim( F, 2 );

else
    patches = extract_patches_from_image( single(I), boxes, ...
        canonical_patchsize, PARAM.Patch_Padding );
    F = feat_func( patches );
end

end
