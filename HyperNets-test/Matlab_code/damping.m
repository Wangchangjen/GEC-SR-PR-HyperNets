function [out] = damping(old,new,m,ii)
% compute the dumping of b and c by a factor m
ii=1;
out = m^ii .* old + (1 - m^ii) .* new;

end

