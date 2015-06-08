function ps = gp_with_measurement( test_x,train_x,train_y, best_score, measure_func, gp_sr )

try
    [m, s2] = gp_sr(test_x,train_x,train_y); 
    ps = measure_func(m,s2,best_score);
catch
    ps = -1e12;
end
    
end
