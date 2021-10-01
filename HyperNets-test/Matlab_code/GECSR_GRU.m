function [MSE_ini,currentResid,currentMeasurementError,MSE,x_hat1,x_hat2,r1_x,r1_z,Q_1_x,Q_1_z,Q_2_x,Q_2_z,data] = GECSR_GRU(y,x,sigma2,initial,z,Nt,Mt,t_max,singular_value, gOut, gxIn,A,C,U,V,SNR_in)
weights = load('HyperNet_GRU_attention'); % GRU_attention-modle
d_k_size = 10;
%%
r1_x=zeros(Mt,1);
r1_z=zeros(Mt,1);% inilization 
j=sqrt(-1);
M =Mt/C; % cluster M
R_2z = zeros(M,C);
R_1x=zeros(Nt,C);
V_1x=ones(Nt,C);
R_2x=zeros(Nt,C);
V_2x=ones(Nt,C);
n_C = Mt/C;

if Nt <= n_C
    n_Ct = Nt/C;
else
    n_Ct = n_C;
end


cluster_tab = [1:n_C:Mt;n_C:n_C:Mt].';
DD = singular_value.^2;



%%
eta1_z=zeros(M,C);
z_hat1=zeros(M,C);
z_hat2=zeros(M,C);
eta2_z=zeros(M,C);
x_hat2=zeros(Nt,C);
eta2_x=zeros(Nt,C);

%% eig_rho < 0.95
yy = abs(y).^2;
delta = Mt/Nt;
ymean = mean(yy);

yy = yy/ymean;
yplus = max(yy,0);
T = (yplus-1)./(yplus+sqrt(delta)-1); 
T = T*ymean;

Yfunc = @(a) 1/Mt*A'*(T.*(A*a));
opts.isreal = false;
[xii,~] = eigs(Yfunc, Nt, 1, 'lr', opts);
u = abs(A*xii).*abs(y);
l = abs(A*xii).*abs(A*xii);
s = norm(u(:))/norm(l(:));
xii = xii*s;

for iter_inial=1:3
    z_ob = A*xii;
    z_abs = abs(y).*sign(z_ob);
    xii = pinv(A)*z_abs;
    MSE_ini(iter_inial) = mean(abs(sign(xii'*x).*xii - x).^2);
end


%% (1)
V_xi = C;
for c=1:C
Pz(c) = mean(DD(:,c))*min(Nt/M,1);
end
if Pz < 1
    V_zi = (1./(Pz));
else
    V_zi = (1./Pz);
end

V_1x = V_xi*ones(Nt,C);
R_1x = repmat(xii,1,C);
V_2x = V_xi*ones(Nt,C);
R_2x = repmat(xii,1,C);
V_1z = bsxfun(@times, V_zi,ones(M,C));
zz = reshape(A*xii+0*crandn(Mt,1), [M C]);
R_1z= zz;
z = reshape(z,[M C]);

R_1z = reshape(A*xii,[M C]);
R_2z = reshape(A*xii,[M C]);
V_2z = V_1z;
Input_s = (singular_value/norm(singular_value));
if length(singular_value) ~= 100
    if length(singular_value) == 200
        Input_s = Input_s(1:2:Nt);
    else
         Input_s = interp(Input_s,2);
    end  
end

D_in = [Input_s.',sqrt(sqrt(SNR_in))];

damp_factor1 = zeros(1,t_max);
damp_factor2 = zeros(1,t_max);
h_z = zeros(1,100+4);
h_x = zeros(1,100+4);
%%  Iteration
for ii=1:t_max
for iter_A=1:C
    %% 1)Compute the posterior mean and covariance of z
    [z_hat_t, Q_1_z] = gOut.estim(real(R_1z(:,iter_A))+j*imag(R_1z(:,iter_A)), 1./real(V_1z(:,iter_A)),iter_A);
    %% 1) Compute the extrinsic information of z
    eta1_z(:,iter_A) =real(mean(Q_1_z))*ones(M,1); %EP-U
    z_hat1(:,iter_A)=z_hat_t;
end
eta1_zz =mean(mean(eta1_z));
gamma_2_z_new = abs(bsxfun(@rdivide,eta1_z, bsxfun(@minus,1,eta1_z.*V_1z)));
r2_z_new=bsxfun(@times,bsxfun(@minus,bsxfun(@rdivide,z_hat1,eta1_z), bsxfun(@times,V_1z,R_1z)),gamma_2_z_new);

if ii == 1
    D_in2 = [1,1,gamma_2_z_new(1)];
elseif ii == 2
    D_in2 = [1,damp_factor1(ii-1),gamma_2_z_new(1)];
else
    D_in2 = [damp_factor1(ii-2),damp_factor1(ii-1),gamma_2_z_new(1)];
end

Inputx = [D_in,D_in2];

[damp_factor1(ii), h_z ]= GRU_att_damp(Inputx, h_z,weights,d_k_size);



V_2z = damp_factor1(ii)*V_2z + (1-damp_factor1(ii))*(1./gamma_2_z_new);
R_2z = damp_factor1(ii)*R_2z + (1-damp_factor1(ii))*(r2_z_new);
%%
for iter_A=1:C
   Q_2_x = inv(diag(V_2x(:,iter_A)) + A(cluster_tab(iter_A,1):cluster_tab(iter_A,2),:)'* diag(V_2z(:,iter_A))*A(cluster_tab(iter_A,1):cluster_tab(iter_A,2),:));
   x_hat2(:,iter_A)=Q_2_x*(R_2x(:,iter_A).*V_2x(:,iter_A)+A(cluster_tab(iter_A,1):cluster_tab(iter_A,2),:)'*(R_2z(:,iter_A).*V_2z(:,iter_A)));% estimation from  linear space :Z=AX
   eta2_x(:,iter_A)  =max(real(mean(diag(Q_2_x)))*ones(Nt,1),eps);
end

gamma_1_x_new = bsxfun(@rdivide,eta2_x, bsxfun(@minus,1,eta2_x.*V_2x));
r1_x_new=bsxfun(@times,bsxfun(@minus,bsxfun(@rdivide,x_hat2,eta2_x), bsxfun(@times,V_2x,R_2x)),gamma_1_x_new);
V_1x=  1./gamma_1_x_new;
R_1x= r1_x_new ;

MRCvx = 1./V_1x;
MRCx = r1_x_new;
[x_hat1, Q_1_x] = gxIn.estim(MRCx, MRCvx) ;
Q_1_x=max(Q_1_x,eps);
eta1_x = real(sum(Q_1_x))/Nt*ones(Nt,1);
gamma_2_x_new =bsxfun(@rdivide, eta1_x,(1-bsxfun(@times,eta1_x,V_1x )));
r2_x_new = bsxfun(@times, bsxfun(@minus,x_hat1./eta1_x,bsxfun(@times,V_1x,R_1x)) ,gamma_2_x_new);

if ii == 1
    D_in2 = [1,1,gamma_2_x_new(1)];
elseif ii == 2
    D_in2 = [1,damp_factor2(ii-1),gamma_2_x_new(1)];
else
    D_in2 = [damp_factor2(ii-2),damp_factor2(ii-1),gamma_2_x_new(1)];
end

Inputx = [D_in,D_in2];

[damp_factor2(ii), h_x ]= GRU_att_damp(Inputx, h_x,weights,d_k_size);

if  ii >=10
    damp_factor2(ii) = 0.1;
end


V_2x = damp_factor2(ii)*V_2x + (1-damp_factor2(ii))*(1./gamma_2_x_new);
R_2x = damp_factor2(ii)*R_2x + (1-damp_factor2(ii))*(r2_x_new);



for iter_A=1:C    
    %% Compute the mean and covariance of z from the linear space
[Q_2_x] = inv_lemma(V_2x(:,iter_A),V_2z(:,iter_A),A(cluster_tab(iter_A,1):cluster_tab(iter_A,2),:),DD(:,iter_A),U(iter_A));
x_hat2(:,iter_A)=Q_2_x*(R_2x(:,iter_A).*V_2x(:,iter_A)+A(cluster_tab(iter_A,1):cluster_tab(iter_A,2),:)'*(R_2z(:,iter_A).*V_2z(:,iter_A)));
z_hat2(:,iter_A)=A(cluster_tab(iter_A,1):cluster_tab(iter_A,2),:)*x_hat2(:,iter_A);  % estimation from  linear space :Z=AX
Q_2_z = real(sum(DD(:,iter_A)./(V_2z(1,iter_A).*DD(:,iter_A)+V_2x(1,iter_A))))/M;
    %% Compute the extrinsic information of z
      eta2_z(:,iter_A) =max(real(mean(diag(Q_2_z)))*ones(M,1),eps); %EP-U
end

gamma_1_z_new = bsxfun(@rdivide,eta2_z, bsxfun(@minus,1,eta2_z.*V_2z));
r1_z_new=bsxfun(@times,bsxfun(@minus,bsxfun(@rdivide,z_hat2,eta2_z), bsxfun(@times,V_2z,R_2z)),gamma_1_z_new);
V_1z= damping( V_1z, 1./gamma_1_z_new, 0,ii) ;
R_1z= damping(R_1z, r1_z_new, 0,ii) ; 
   
%% MMSE calculation
x_hat1=sign(x_hat1'*x).*x_hat1; % for complex signal
MSE(ii)=(norm(x-x_hat1)^(2))/Nt;% MSE
currentMeasurementError(ii)=norm(abs(A*x_hat1)-initial.abs_y)/norm(initial.abs_y);
currentResid(ii) = mean(eta1_x);
data.xhat1(:,ii) = x_hat1;
end
end
 

