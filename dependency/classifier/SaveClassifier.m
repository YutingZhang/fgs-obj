function SaveClassifier( classifier_model_all, Classifier_Type, ...
    CLASSIFIER_DIR, CategOfInterest, SUB_DIR )

if ~exist('SUB_DIR','var')
    SUB_DIR = '';
end

if isempty( SUB_DIR )
    fprintf( 1, 'Save model : ' );
else
    fprintf( 1, 'Save model %s: ', SUB_DIR );
end

mkdir_p( fullfile( CLASSIFIER_DIR, SUB_DIR ) );

for c = 1:length(CategOfInterest)
    fprintf( 1, '%s, ', CategOfInterest{c} );
    classifier_model_c = struct( ...
        'scale', {classifier_model_all.scale}, ...
        'w',     {classifier_model_all.w(c,:)}, ...
        'bias',  {classifier_model_all.bias(c)}, ...
        'type',  {Classifier_Type} );
    CLASSIFIER_PATH = fullfile( CLASSIFIER_DIR, SUB_DIR, [CategOfInterest{c} '.mat'] );
    save( CLASSIFIER_PATH, '-struct', 'classifier_model_c' );
end
fprintf( 1, '\n' );

end
