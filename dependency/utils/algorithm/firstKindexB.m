function chosenIdxB = firstKindexB( a, k, varargin )

if iscell(a)
    if isscalar(k)
        chosenIdxB = cellfun( @(a1) {firstKindexB_single(a1, k, varargin{:})}, a );
    else
        chosenIdxB = arrayfun( @(a1,k1) {firstKindexB_single(a1{1}, k1, varargin{:})}, a, k );
    end
else
    chosenIdxB = firstKindexB_single( a, k, varargin{:} );
end

end

function chosenIdxB = firstKindexB_single( a, k, varargin )

assert( isvector(a), 'a must be a vector' );

if k>length(a)
    chosenIdxB = true(size(a));
else
    [~,ord] = sort(a,varargin{:});
    chosenIdxB = false(size(a));
    chosenIdxB( ord(1:min(k,end)) ) = true;
end

end
