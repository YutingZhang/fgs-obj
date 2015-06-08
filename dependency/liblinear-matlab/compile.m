
CXXFLAGS='-Wall -Wconversion -O3 -fPIC';
MEX_OPTION=[ '-largeArrayDims ' ...
    sprintf( '-I%s ', getenv('MATLAB') ), ...
    sprintf('CFLAGS="%s" ',CXXFLAGS) ...
    sprintf('CXXFLAGS="%s" ',CXXFLAGS) ];

eval( [ 'mex ' MEX_OPTION 'matlab/libsvmread.cpp' ] );
eval( [ 'mex ' MEX_OPTION 'matlab/libsvmwrite.cpp' ] );
eval( [ 'mex ' MEX_OPTION '-c ' 'linear.cpp' ] );
eval( [ 'mex ' MEX_OPTION '-c ' 'tron.cpp' ] );
eval( [ 'mex ' MEX_OPTION '-c ' 'matlab/linear_model_matlab.cpp' ] );
eval( [ 'mex ' MEX_OPTION '-lmwblas ' ...
        'matlab/liblinear_train.cpp '   'linear.o tron.o linear_model_matlab.o' ] );
eval( [ 'mex ' MEX_OPTION '-lmwblas ' ...
        'matlab/liblinear_predict.cpp ' 'linear.o tron.o linear_model_matlab.o' ] );

delete *.o

