function [Parent_Dir, has_parent, Index_In_Parent, Child_Shape] = ...
    sysParentLink( Specific_Dir, min_Child_Dim )

persistent parent_cache % N x 3: 1 - filename, 2 - datenum, 3 - content

if ~exist('min_Child_Dim', 'var') || isempty(min_Child_Dim)
    min_Child_Dim = 1;
end

PARENT_TXT_PATH = fullfile( Specific_Dir, 'parent.txt' );

if exist( PARENT_TXT_PATH, 'file' )

    need_refresh = 1;
    if isempty(parent_cache)
        parent_idx = 1;
    else

        [~,parent_idx] = ismember( PARENT_TXT_PATH , parent_cache(:,1) );
        if parent_idx > 0
            old_datenum = parent_cache{parent_idx,2};
            f = dir(PARENT_TXT_PATH);
            if f.datenum<=old_datenum
                need_refresh = 0;
            end
        else
            parent_idx = size( parent_cache,1 ) + 1;
        end

    end

    if need_refresh

        fid = fopen( PARENT_TXT_PATH, 'r' );
        tline0 = fgetl(fid);
        tline1 = fgetl(fid);
        tline2 = fgetl(fid);
        fclose(fid);
        Child_Shape = sscanf(tline0,'%d');
        Parent_Dir = fullfile(Specific_Dir,tline1);
        Index_In_Parent = sscanf(tline2,'%d');
        
        parent_cache{parent_idx,5} = Child_Shape;
        parent_cache{parent_idx,4} = Index_In_Parent;
        parent_cache{parent_idx,3} = Parent_Dir;
        parent_cache{parent_idx,1} = PARENT_TXT_PATH;
        f = dir(PARENT_TXT_PATH);
        parent_cache{parent_idx,2} = f.datenum;

    else

        Parent_Dir      = parent_cache{parent_idx, 3};
        Index_In_Parent = parent_cache{parent_idx, 4};
        Child_Shape     = parent_cache{parent_idx, 5};
        
    end
    
    Index_In_Parent(length(Index_In_Parent)+1:min_Child_Dim) = 1;
    Child_Shape(length(Child_Shape)+1:min_Child_Dim) = 1;

    has_parent = true;
    
else
    
    has_parent = false;
    Parent_Dir   = Specific_Dir;
    Index_In_Parent = ones(1,min_Child_Dim);
    Child_Shape  = ones(1,min_Child_Dim);
    
end


end
