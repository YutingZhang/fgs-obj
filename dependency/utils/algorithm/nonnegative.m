function A=nonnegative(A)
A=max(A,eval([class(A) '(0)']));
end
