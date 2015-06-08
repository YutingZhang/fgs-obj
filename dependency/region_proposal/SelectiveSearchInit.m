function SelectiveSearchInit()

CurrentDir = pwd;
[SSDir,~,~] = fileparts( which('Image2HierarchicalGrouping') );

fp  = which('anigauss');
[~,~,ext] = fileparts( fp );
if( isempty(fp) || ~strncmp(ext,'.mex',4) )
    fprintf(1,'Compile anigauss\n');
    cd( SSDir );
    mex Dependencies/anigaussm/anigauss_mex.c Dependencies/anigaussm/anigauss.c -output anigauss
end

fp  = which('mexCountWordsIndex');
[~,~,ext] = fileparts( fp );
if( isempty(fp) || ~strncmp(ext,'.mex',4) )
    fprintf(1,'Compile mexCountWordsIndex\n');
    cd( SSDir );
    mex Dependencies/mexCountWordsIndex.cpp
end

% Compile the code of Felzenszwalb and Huttenlocher, IJCV 2004.
fp  = which('mexFelzenSegmentIndex');
[~,~,ext] = fileparts( fp );
if( isempty(fp) || ~strncmp(ext,'.mex',4) )
    fprintf(1,'Compile mexFelzenSegmentIndex\n');
    cd( SSDir );
    mex Dependencies/FelzenSegment/mexFelzenSegmentIndex.cpp -output mexFelzenSegmentIndex;
end

cd( CurrentDir );

end
