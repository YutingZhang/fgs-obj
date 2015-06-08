function [Redirected_Dirs, is_redirected, Redirection_Shape] = ...
    sysRedirectList( Specific_Dir, min_Redirection_Dim )

persistent redirect_cache % N x 3: 1 - filename, 2 - datenum, 3 - content

if ~exist('Min_Redirection_Dim', 'var') || isempty(min_Redirection_Dim)
    min_Redirection_Dim = 1;
end

REDIRECT_LIST_PATH = fullfile( Specific_Dir, 'redirect.list' );

if exist( REDIRECT_LIST_PATH, 'file' )

    need_refresh = 1;
    if isempty(redirect_cache)
        redirect_idx = 1;
    else

        [~,redirect_idx] = ismember( REDIRECT_LIST_PATH , redirect_cache(:,1) );
        if redirect_idx > 0
            old_datenum = redirect_cache{redirect_idx,2};
            f = dir(REDIRECT_LIST_PATH);
            if f.datenum<=old_datenum
                need_refresh = 0;
            end
        else
            redirect_idx = size( redirect_cache,1 ) + 1;
        end

    end

    if need_refresh

        Redirected_Dirs = {};
        fid = fopen( REDIRECT_LIST_PATH, 'r' );
        tline = fgetl(fid);
        if tline(1) == '#'
            Redirection_Shape = str2num(tline(2:end));
            tline = fgetl(fid);
        else
            Redirection_Shape = [];
        end
        while ischar(tline)
            if ~isempty(strtrim(tline))
                Redirected_Dirs = [Redirected_Dirs;{fullfile(Specific_Dir,tline)}];
                tline = fgetl(fid);
            end
        end
        fclose(fid);

        if isempty( Redirection_Shape )
            Redirection_Shape = length(Redirected_Dirs);
        end
        
        redirect_cache{redirect_idx,4} = Redirection_Shape;
        redirect_cache{redirect_idx,3} = Redirected_Dirs;
        redirect_cache{redirect_idx,1} = REDIRECT_LIST_PATH;
        f = dir(REDIRECT_LIST_PATH);
        redirect_cache{redirect_idx,2} = f.datenum;

    else

        Redirected_Dirs   = redirect_cache{redirect_idx, 3};
        Redirection_Shape = redirect_cache{redirect_idx, 4};
        
    end
    
    Redirection_Shape(length(Redirection_Shape)+1:min_Redirection_Dim) = 1;

    is_redirected = true;
    
else
    
    is_redirected = false;
    Redirected_Dirs   = {Specific_Dir};
    Redirection_Shape = ones(1,min_Redirection_Dim);
    
end


end
