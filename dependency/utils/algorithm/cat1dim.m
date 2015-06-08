function A = cat1dim( C, dim )

if ~exist('dim','var')
    dim = 1;
end

s = size(C);
L = arrayfun( @(a) {ones(a,1)}, s );
L{dim} = s(dim);

A = mat2cell( C, L{:} );
A = cellfun( @(a) {cat(dim,a{:})}, A );

end
