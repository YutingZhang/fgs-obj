function gp_setup

if ismac
    fprintf( 'gp_setup: I do not want to compile on mac\n' );
    return;
end

CUR_PATH = pwd;
GP_PATH = fileparts( which('gp') );

cd( fullfile(GP_PATH, 'util') );

% compile solve_chol
% if ismac && ~exist('OCTAVE_VERSION')
%     % patch for matlab on mac
%     mex -O -Dchar16_t=UINT16_T -lmwlapack solve_chol.c  
% else
    make
% end

% compile lbfgsb
cd( fullfile(GP_PATH, 'util/lbfgsb') );

in_fid = fopen( 'Makefile', 'r' );
out_fid = fopen( 'Makefile.machine', 'w' );
tline = fgetl( in_fid );
while ischar(tline)
    if ~isempty( regexp( tline, '[\t ]*\<MATLAB_HOME\>[\t ]*=', 'once' ) )
        tline=sprintf('MATLAB_HOME=%s',matlabroot);
%     elseif ~isempty( regexp( tline, '[\t ]*\<CFLAGS\>[\t ]*=', 'once' ) )
%         if ismac
%             tline=[tline ' -Dchar16_t=UINT16_T'];
%         end
    elseif ~isempty( regexp( tline, '[\t ]*\<MEX_SUFFIX\>[\t ]*=', 'once' ) )
        tline=sprintf('MEX_SUFFIX=%s',mexext);
    elseif ~isempty( regexp( tline, '[\t ]*\<MATLAB_LIB\>[\t ]*=', 'once' ) )
        tline=sprintf('MATLAB_LIB=-L$(MATLAB_HOME)/bin/%s',computer('arch'));
    end

    fprintf( out_fid, '%s\n', tline );
    tline = fgetl( in_fid );
end
fclose(out_fid);
fclose(in_fid);

!make -f Makefile.machine
delete Makefile.machine

%

cd( CUR_PATH );


end
