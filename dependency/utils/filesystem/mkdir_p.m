function mkdir_p( a )

if ~exist( a, 'dir' )
    mkdir(a);
end

end
