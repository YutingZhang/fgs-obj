% FeatureNorm4Train

FEAT_MEAN_NORM_PATH = fullfile( SPECIFIC_DIRS.FeatureNorm4Train, 'mean_feat_norm.mat' );

if exist(FEAT_MEAN_NORM_PATH,'file')
    fprintf( 1, 'Already Exists\n' );
else
    mkdir_p( SPECIFIC_DIRS.FeatureNorm4Train );
    fprintf(1, 'Loading data list: '); tic
    TRAIN_DATA_LIST = GetDataList( PARAM.Train_DataSet );
    imageN = length(TRAIN_DATA_LIST);
    toc
    fprintf( 1, 'Compute feature norm: ' ); tic
    Fnorm = cell(imageN,1);
    for k = 1:imageN
        Fnorm{k} = GetProposedFeatureNorm( PARAM.Train_DataSet, TRAIN_DATA_LIST(k).im );
    end
    Fnorm = cat( 2, Fnorm{:} );
    mean_feat_norm = nan(size(Fnorm,1),1);
    median_feat_norm = nan(size(Fnorm,1),1);
    for t = 1:size(Fnorm,1)
        mean_feat_norm(t)   = mean( Fnorm( t, ~( isinf(Fnorm(t,:)) | isnan(Fnorm(t,:)) ) ) );
        median_feat_norm(t) = median( Fnorm( t, ~( isinf(Fnorm(t,:)) | isnan(Fnorm(t,:)) ) ) );
    end

    % if there is some extremely large values
    if any(mean_feat_norm./median_feat_norm(t)>4)
        mean_feat_norm = median_feat_norm; % use median instead
    end
    
    save( FEAT_MEAN_NORM_PATH, 'mean_feat_norm' );
    clear Fnorm
    toc
    fprintf( 1, 'Done\n' );
end
