function url2file( URL, FILE_PATH, WGET_OPTIONS )

if ~exist( 'WGET_OPTIONS', 'var' )
    WGET_OPTIONS = '';
end

status = system( 'wget --version' );

if status % no wget
    tic
    fprintf( 1, 'Use urlwrite : ' );
    urlwrite( URL, FILE_PATH );
    toc
else % have wget
    WGET_CMD = sprintf('wget %s -O ''%s'' ''%s''', WGET_OPTIONS, FILE_PATH, URL );
    system( WGET_CMD );
end

end
