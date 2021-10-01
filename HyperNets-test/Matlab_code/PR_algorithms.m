function [MSE_RWF, MSE_TAF, MSE_HIO] = PR_algorithms(y,x,sigma2,initial,z,Nt,Mt,t_max,singular_value, gOut, gxIn,A,C,U,V,SNR_in)

opts1 = struct;
opts1.initMethod = 'weightedspectral';
opts1.algorithm = 'RWF';
opts1.isComplex = true;
opts1.tol = 1e-10;
opts1.verbose = 1;



opts2 = struct;
opts2.initMethod = 'AmplitudeSpectral';
opts2.algorithm = 'TAF';
opts2.isComplex = true;
opts2.tol = 1e-10;
opts2.verbose = 1;


opts3 = struct;
opts3.initMethod = 'Truncatedspectral';
opts3.algorithm = 'Fienup';
opts3.isComplex = true;
opts3.FienupTuning = 0.5;
opts3.tol = 1e-6;
opts3.verbose = 1;

[x_hat1, outs1, opts1] = solvePhaseRetrieval(A, A', initial.abs_y, Nt, opts1); % RWF
[x_hat2, outs2, opts2] = solvePhaseRetrieval(A, A', initial.abs_y, Nt, opts2); % TAF
[x_hat3, outs3, opts3] = solvePhaseRetrieval(A, A', initial.abs_y, Nt, opts3); % HIO


%% MMSE calculation
x_hat1 = sign(x_hat1'*x).*x_hat1; % for complex signal
x_hat2 = sign(x_hat2'*x).*x_hat2;
x_hat3 = sign(x_hat3'*x).*x_hat3;

MSE_RWF=(norm(x-x_hat1)^(2))/Nt;% MSE
MSE_TAF=(norm(x-x_hat2)^(2))/Nt;% MSE
MSE_HIO=(norm(x-x_hat3)^(2))/Nt;% MSE
end
 

