function out = tanh_f(input)
out = (exp(input) - exp(-1.*input))./(exp(input) + exp(-1.*input));
end