function compile_dependency

% gpml
[~, ~, ext1__] = fileparts( which( 'solve_chol' ) );
[~, ~, ext2__] = fileparts( which( 'lbfgsb' ) );
if ~strncmp(ext1__,'.mex',length('.mex')) ||  ~strncmp(ext2__,'.mex',length('.mex'))
    gp_setup
    fprintf(1,'gpml toolbox is compiled\n');
end
clear ext1__ ext2__

% liblinear
liblinear_init

% minFunc
MIN_FUNC_ROOT_    = fileparts( which( 'minFunc' ) );
MIN_FUNC_MEX_FILES_ = dir( fullfile(MIN_FUNC_ROOT_,'mex/*.c') );
MIN_FUNC_MEX_FILES_ = {MIN_FUNC_MEX_FILES_.name};

MIN_FUNC_COMPILE_FIRST_FILE_ = 0;
for k__ = 1:length( MIN_FUNC_MEX_FILES_ )
    [~,MIN_FUNC_FN_,~] = fileparts( MIN_FUNC_MEX_FILES_{k__} );
    if exist(MIN_FUNC_FN_,'file') ~= 3 
        if ~MIN_FUNC_COMPILE_FIRST_FILE_
            mkdir_p( fullfile(MIN_FUNC_ROOT_,'compile') );
            addpath( fullfile(MIN_FUNC_ROOT_,'compile') );
            fprintf( 1, 'Compile minFunc : \n' );
            MIN_FUNC_COMPILE_FIRST_FILE_ = 1;
        end
        mex( '-O', '-outdir', fullfile(MIN_FUNC_ROOT_,'compile'), ...
            fullfile(MIN_FUNC_ROOT_,'mex/', MIN_FUNC_MEX_FILES_{k__} ) );
    end
end

if MIN_FUNC_COMPILE_FIRST_FILE_
    fprintf( 'minFunc dependencies are compiled\n' );
end
clear MIN_FUNC_ROOT_ MIN_FUNC_MEX_FILES_ MIN_FUNC_COMPILE_FIRST_FILE_ MIN_FUNC_FN_ k__


end
