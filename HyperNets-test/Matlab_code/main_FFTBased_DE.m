
clc
clear all
warning('off','all') % wearming close

SNR_dB = 25;%  SNR(dB)
N=100;%dimension of x
M=4*N ; %dimension of observation v
lamda =0.5; %sparasity
tr=1;% number of realizations
t_max=20; %number of iteration
j=sqrt(-1);
eig_max = 1;
eig_min = 0;
eig_rho = 0.98;
% load('seed_data.mat')
% rng(scurr);
%% Decentralized groups
C=1;
n_C = M/C;

if N <= n_C
    n_Ct = N;
else
    n_Ct = n_C;
end

cluster_tab = [1:n_C:M;n_C:n_C:M].';

for jj=1:tr
    A = crandn(M,N)/sqrt(2*M); %GS  
    [u , d, v] = svd(A);
    eig_num = min(M,N);
%     singular_value =  diag(d);
    singular_value =  eig_max.*eig_rho.^[1:1:(eig_num)];
    singular_value = sort(singular_value(:),'descend')/norm(singular_value)...
        * sqrt(M)/sqrt(10^(-0.1*SNR_dB)); 

    
    %% bernoulli input CAWGN
    bernoulli=rand(N,1)>(1-lamda);% bernoulli distribution
    Gauss=sqrt(1/(2*lamda))*(randn(N,1)+1i*randn(N,1));% complex  Gaussian  distribution
    x=(bernoulli.*Gauss);%  x   
    
    indx_big = find( abs(x) > 0.1*sqrt(mean(abs(x).^2)) );
    initial.xmean0 = mean(x(indx_big));            % mean of big elements
    initial.xvar0 = var(x(indx_big)); 
    initial.lamda =lamda;
    %---Output estimation function for sparse signal---%
    gxIn = CAwgnEstimIn(0,1/initial.lamda, initial.lamda);

    A_all_U=Unitary(M);
    A_all_V=Unitary(N);

    if M >= N
        A = A_all_U*[diag(singular_value);zeros(M-N,N)]*A_all_V;
    else
        A = A_all_U*[diag(singular_value),zeros(M,N-M)]*A_all_V;
    end

    Frob_norm_A = norm(A,'fro').^2;
    z=A*x;
    
    U = UnitaryOperator(A_all_U);
    V =  UnitaryOperator(A_all_V);
    svd_eig =singular_value;
    
    SNR = 10.^(SNR_dB/10);% SNR
    sigma2 = 10^(-0.1*SNR_dB)*mean(abs(z).^2);

    noise=sqrt(sigma2/2)*(randn(M,1)+1i*randn(M,1));% noise(complex Gaussian distribution)
    y = z + noise; % observation
    
    %---Output estimation function for PR---%
   initial.abs_y = abs(y);
   gOut=ncCAwgnEstimOut(initial.abs_y, sigma2*ones(n_C,1),cluster_tab);
   
    %% Run GEC-SR with HyperNets and algorithms
    for testtime=1:1
     % run HyperNet 
     [~,~, ~,MSE_mtx_GLM1(:,jj),x_est1,x_hat1,r1_x1,r1_z1,Q_1_x1,Q_1_z1,Q_2_x1,Q_2_z1,~]= GECSR_Hyper(y,x,sigma2,initial,z,N,M,t_max, svd_eig, gOut, gxIn,A,C,U,V,SNR); 
     % run HyperGRU
     [~,~, ~,MSE_mtx_GLM2(:,jj),x_est2,x_hat2,r1_x2,r1_z2,Q_1_x2,Q_1_z2,Q_2_x2,Q_2_z2,~]= GECSR_GRU(y,x,sigma2,initial,z,N,M,t_max, svd_eig, gOut, gxIn,A,C,U,V,SNR); 
     % run RWF & TAF & HIO 
     [MSE_RWF(:,jj),MSE_TAF(:,jj), MSE_HIO(:,jj)]= PR_algorithms(y,x,sigma2,initial,z,N,M,t_max, svd_eig, gOut, gxIn,A,C,U,V,SNR); % run RWF & TAF & HIO    
    end
end

%%
figure(1)
MSE_GLM1=mean(MSE_mtx_GLM1,2); % average over tr realization for HyperNet
MSE_GLM2=mean(MSE_mtx_GLM2,2); % average over tr realization for HyperGRU

MSE_GLM_RWF=mean(MSE_RWF,2); % average over tr realization for RWF
MSE_GLM_TAF=mean(MSE_TAF,2); % average over tr realization for TAF
MSE_GLM_HIO=mean(MSE_HIO,2); % average over tr realization for HIO

plot(1:t_max,10*log10(MSE_GLM1),'-ro');
hold on
plot(1:t_max,10*log10(MSE_GLM2),'-go');
plot(1:t_max,ones(t_max,1).*10*log10(MSE_GLM_RWF),'-bo');
plot(1:t_max,ones(t_max,1).*10*log10(MSE_GLM_TAF),'-r sq');
plot(1:t_max,ones(t_max,1).*10*log10(MSE_GLM_HIO),'-bx');




