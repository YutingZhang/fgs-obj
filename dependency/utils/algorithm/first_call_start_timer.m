classdef first_call_start_timer < handle
    properties ( GetAccess = public, SetAccess = private )
        start_time = []; 
        elapsed_time
    end
    methods
        function value = get.elapsed_time(obj)
            if isempty(obj.start_time)
                obj.start_time = tic;
                value = 0;
            else
                value = toc(obj.start_time);
            end
        end
        function reset(obj)
            obj.start_time = [];
        end
    end
end
