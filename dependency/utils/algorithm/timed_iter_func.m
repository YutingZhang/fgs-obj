function wrapped_func = timed_iter_func( time_out, func )
% wrapped_func = timed_iter_func( timeout, func )

tinfo.ftimer   = first_call_start_timer;
tinfo.time_out = time_out;

wrapped_func = @(varargin) timed_iter_func_wrap( tinfo, func, varargin{:} );

end

function varargout = timed_iter_func_wrap( tinfo, func, varargin )

if tinfo.ftimer.elapsed_time>tinfo.time_out
    error( 'timed_iter_func:timeout', 'timed_obj is timed out.' );
end

varargout      = cell(1, nargout);
[varargout{:}] = func( varargin{:} );

end

