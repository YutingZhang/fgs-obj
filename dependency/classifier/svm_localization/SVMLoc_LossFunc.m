function delta = SVMLoc_LossFunc(param, y, yhat)

if y.label>0
    if y.overlap ~=1
        error( 'y.overlap (y is a pos groundtruth) must be 1' );
    end
    if yhat.label>0
        delta = SVMLoc_OverlapLoss( param, yhat.overlap );
    else
        delta = 1;
    end
    delta = delta*param.pos_weight;
else
    delta = double( y.label ~= yhat.label );
end

end
