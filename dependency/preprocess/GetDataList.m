function DATA_LIST = GetDataList( list_type, PrepDataset_SpecificDir, DATASET_DIR )

if ( ~exist( 'PrepDataset_SpecificDir', 'var' ) || isempty(PrepDataset_SpecificDir) ) ...
        && evalin( 'caller', 'exist(''SPECIFIC_DIRS'',''var'')' )
    PrepDataset_SpecificDir = evalin( 'caller', 'SPECIFIC_DIRS.PrepDataset' );
end

if ~exist('DATASET_DIR','var') % || isempty(DATASET_DIR)
    if evalin( 'caller', 'exist(''SPECIFIC_DIRS'',''var'')' ) && evalin( 'caller', 'isfield(SPECIFIC_DIRS,''VOC2007_ROOT'')' )
        DATASET_DIR = evalin( 'caller', 'SPECIFIC_DIRS.VOC2007_ROOT' );
        DATASET_DIR = fullfile( DATASET_DIR, 'JPEGImages' );
    else
        DATASET_DIR = '';
    end
end

LIST_FILE_NAME = fullfile( PrepDataset_SpecificDir, 'list', [list_type, '.mat'] );
MAT_CONTENT = load(LIST_FILE_NAME);
DATA_LIST = MAT_CONTENT.DATA_LIST;

if ~isempty(DATASET_DIR)
    for k = 1:length(DATA_LIST)
        [~,fn,ext] = fileparts( DATA_LIST(k).im );
        DATA_LIST(k).im = fullfile(DATASET_DIR,[fn ext]);
    end
end

end
