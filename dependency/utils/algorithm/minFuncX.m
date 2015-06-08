function varargout = minFuncX(funObj,x0,options,varargin)
% functions the same as minFunc, but extended as follows:
% 1) It accepts any kind of x0 (including cell and struct)
% 2) It supports timeout (in second), timeout_handler ('error', 'stop')

if ~exist( 'options', 'var' )
    options = [];
end

% set up objective function for nonstardard input
nonstandardInput = ~isvector(x0) || ~isnumeric(x0);
if nonstandardInput
    funObj_1 = @(x,varargin) UnwrappedObjFunc( funObj, x0, x, varargin ); 
    x0_1 = unwrap2vec(x0);
else
    funObj_1 = funObj;
    x0_1 = x0;
end

% timeout
get_value_when_timeout = 0;
if isstruct( options ) && isfield( options, 'timeout' ) && options.timeout < inf
    if isfield( options, 'timeout_handler' )
        if isempty(options.timeout_handler)
            options.timeout_handler = 'error';
        end
    else
        options.timeout_handler = 'error';
    end
    switch options.timeout_handler
        case 'error'
            funObj_2 = timed_iter_func( options.timeout, funObj_1);
        case 'stop'
            get_value_when_timeout = 1;
            bestH = handleObj( struct('f', inf, 'x', x0) );
            funObj_2_1 = timed_iter_func( options.timeout, funObj_1);
            funObj_2   = @(x,varargin) RecordObjFunc( ...
                funObj_2_1, bestH, x, varargin{:} );
        otherwise
            error( 'Unrecognized timeout_handler' );
    end
else
    funObj_2 = funObj_1;
end

% optimization
ARGOUT = cell(1,max(nargout,1));

if get_value_when_timeout
    try 
        [ARGOUT{:}] = minFunc(funObj_2,x0_1,options,varargin{:});
    catch e
        if strcmp(e.identifier,'timed_iter_func:timeout')
            warning( 'optimization timeout' );
            ARGOUT{1} = bestH.data.x;
            ARGOUT{2} = bestH.data.f;
            ARGOUT{3} = -5; %exitflags: use -5 to denote timeout
            ARGOUT{4} = struct();
        else
            rethrow(e);
        end
    end
else
    [ARGOUT{:}] = minFunc(funObj_2,x0_1,options,varargin{:});
end

% set up nonstardard output
if nonstandardInput
    ARGOUT{1} = rewrap_vec( x0, ARGOUT{1} );
end

% output
varargout = ARGOUT;

end

function varargout = RecordObjFunc( funObj, bestH, x, varargin )

ARGOUT = cell(1,max(nargout,1));
[ARGOUT{:}] = funObj( x, varargin{:} );

if ARGOUT{1}<bestH.data.f
    bestH.data.f = ARGOUT{1};
    bestH.data.x = x;
end

varargout = ARGOUT;

end

function varargout = UnwrappedObjFunc( funObj, x0, x, varargin )

if nargout>2
    error( 'Do NOT support 2-order unwrapping' );
end

ARGOUT = cell(1,max(nargout,1));
[ARGOUT{:}] = funObj( rewrap_vec(x0,x), varargin{:} );

if nargout>=2
    ARGOUT{2} = unwrap2vec(ARGOUT{2});
end

varargout = ARGOUT;

end

