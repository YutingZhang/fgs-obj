function hypD = gp_hyp_dim( func, D )

dim_expr = func{1}( func{2:end} );
hypD = eval( dim_expr );

end
