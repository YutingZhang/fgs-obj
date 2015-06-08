function liblinear_init

if ~exist( 'liblinear_train' ) || ~exist( 'liblinear_predict' )
    fprintf(1,'Compile liblinear\n');
    CURRENT_FOLDER = pwd;
    THIS_FOLDER = fileparts(which(mfilename('fullpath')));
    cd(THIS_FOLDER);
    run('compile.m');
    cd(CURRENT_FOLDER);
end

end

