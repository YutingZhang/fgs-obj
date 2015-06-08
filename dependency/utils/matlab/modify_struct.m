function S = modify_struct( S, field_name, field_value )

eval(sprintf('S.%s = field_value;', field_name));

