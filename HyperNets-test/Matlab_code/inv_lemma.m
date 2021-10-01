function [Q_2_x,aa] = inv_lemma(V_2x,V_2z,A,sigular_value,sigular_vector)
% inv(V_2x+A'*V_2z*A)
[a b]=size(A);
U=diag(1./V_2x);
V=diag(1./V_2z);
% tic
% CT = inv(V+A*U*A');
if a >b 
aa = ((1./V_2z)+[sigular_value.*(1./mean(V_2x));zeros(a-b,1)]).^-1;
else
aa = ((1./V_2z)+sigular_value.*(1./mean(V_2x))).^-1;
end
CT3 =sigular_vector.HaarMatrix*bsxfun(@ldivide,1./aa,sigular_vector.tHaarMatrix);
Q_2_x = U-U*A'*CT3*A*U;
% Q_2_x = U-U*A'*CT*A*U;
end