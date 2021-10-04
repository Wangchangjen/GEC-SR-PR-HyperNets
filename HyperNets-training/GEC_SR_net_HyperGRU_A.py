# In[1]: import functions
import tensorflow as tf
import numpy as np
import scipy.io as sc 
from X_ini import Phase_ini2
import math
from numpy import linalg as LA
# In[2] parameters setting
N = 100
M = 4*N
snrdb_train=np.array([20],dtype=np.float64)

snr_train = 10.0 ** (snrdb_train/10.0)  #train_SNR_linear
batch_size= 30
train_size = 120
epochs=10
itermax=10
Constan_itermax = 0

diff_iteration = itermax-Constan_itermax
group_C = 1
obj_rho1 = 0.5 # sparse signal range 0~1

M_c = int(M/group_C)
cluster_tab = np.zeros([group_C,2])

for ii in range(group_C):
    cluster_tab[ii,:] = [ii*M_c,(ii+1)*M_c]

valid_size=4800 # 4800/12000
rho=0.5 #correlation
snrdb_test = snrdb_train
snr_test = 10.0 ** (snrdb_test/10.0)  #train_SNR_linear
sigma2 = 10.**(-snrdb_test/10.)/2
weight_mat= r'C:\Users\adm\Desktop\HyperNets'


# We define eps value to avoid numerical problem.
eps=1e-20*tf.ones(1,dtype='float64') 
# In[3] Define functions

# 1) Amplitude observation: y = abs(z+w)   
def abq_y(y_input):
    abs_y = abs(y_input)
    return abs_y

# 2) Generate data function:  
def generate_data_iid_test(B,M,N,SNR,obj_rho_):  
    B_ = np.int(B/2)
    sigma2e = 10.**(-(SNR+(2*np.random.rand(B_,1)-1)*15)/10.)
        
    # exp distribution
    singular_num = N
    singular_rho = 0.98
    DD_1 = 1*np.power(singular_rho,np.array(range(singular_num))+1)
    DDD_1 = np.sqrt(M)*(DD_1/LA.norm(DD_1))/np.sqrt(sigma2e)
 
    # i.i.d GS
    H1_ = (np.random.randn(B_,M,N)+1j*np.random.randn(B_,M,N))
    U, D, V = LA.svd(H1_, full_matrices=True)
    DDD_2 = np.sqrt(M)*(D/np.expand_dims(LA.norm(D,axis=1,ord=2),-1))/np.sqrt(sigma2e) # i.i.d
    
    H_1 = np.matmul(U[..., :N] * DDD_1[..., None, :], V)
    H_4 = np.matmul(U[..., :N] * DDD_2[..., None, :], V)
    
    H_ = np.concatenate((H_1,H_4),0)
    
    
    # Signal part: Sparse signal (complex)
    bernoulli = np.random.rand(B,N,1) > np.expand_dims(1-obj_rho_,axis = 2)
    Gauss = np.sqrt(1/(2*np.expand_dims(obj_rho_,axis = 2)))*(np.random.randn(B,N,1)+1j*np.random.randn(B,N,1))
    x1_ = bernoulli* Gauss # complex-sprase-gaussian
    z_ = np.matmul(H_,x1_)
    
    # noise (complex)
    w=np.sqrt(1/2)*(np.random.randn(B,M,1)+1j*np.random.randn(B,M,1))
    y_ = z_+w
       
    # y_, H_, z_,x1_ transform complex value to real value:
    x_real = np.concatenate((np.real(x1_),np.imag(x1_)),1)
    H_top = np.concatenate((np.real(H_),np.imag(H_)),1)
    H_low = np.concatenate((-np.imag(H_),np.real(H_)),1)
    H_real = np.concatenate((H_top,H_low),2)
    y_abs = abq_y(y_) 
    
    y_real = np.concatenate((np.real(y_),np.imag(y_)),1)
    z_real = np.concatenate((np.real(z_),np.imag(z_)),1)
      
    x2_M_c = Phase_ini2(y_,B,M,N,y_abs,H_)     # Initial solution of sparse signal  
 
    # Output list:
    x2_out = x2_M_c
    y_abs_out = y_abs
    y_real_out = y_real
    z_real_out = z_real
    x_complex_out = x1_
    x_real_out = x_real

    D_out1 = DDD_1 # exp  
    D_out4 = (DDD_2) # i.i.d
    D_out = np.concatenate((D_out1,D_out4),0) # All D
    SNR_out = 1/np.concatenate((sigma2e,sigma2e),0)   # SNR
    
    return y_abs_out, H_real, x_real_out, x_complex_out, y_real_out, z_real_out , x2_out, D_out,  SNR_out

# 3) average variance
def mean_var(varin): # varin = (Batchsize, Groups,Each group size)
#    t0 = varin.shape[0] # Batchsize
    t1 = varin.shape[1] # Each group size
    varin_mean = tf.reduce_mean(varin,1) # (Batchsize, Groups)
    varin_mean1 = tf.expand_dims(varin_mean,-1) 
    varin_mean2 = tf.matmul(varin_mean1,tf.ones([1,t1],dtype='float64')) # (Batchsize, Groups, Each group size)
    return varin_mean2


# 4) Conjugate complex multiplication based on real value:
def complex_conj_mul_real(a,b,c): # a'*b => c = -1  , a*b => c=1
    real_part = tf.multiply(a,b) 
    a2 = tf.concat([c*a[:,N:2*N],a[:,0:N]],-1) 
    imag_part = tf.multiply(a2,b) 
    return  real_part, imag_part


# 5) disambig phase rotation:
def disambig1Drfft(xhat,x):
    real_part , imag_part= complex_conj_mul_real(xhat,x,-1)
    
    real_part_sum = tf.reduce_sum(real_part,1,keepdims=True)
    imag_part_sum = tf.reduce_sum(imag_part,1,keepdims=True)
    nor_abs = tf.sqrt(tf.square(real_part_sum)+tf.square(imag_part_sum))
    
    real_aa = tf.divide(real_part_sum,nor_abs)
    imag_aa = tf.divide(imag_part_sum,nor_abs)
    
    xout_real = tf.multiply(real_aa, xhat[:,0:N]) - tf.multiply(imag_aa, xhat[:,N:2*N])
    xout_imag = tf.multiply(real_aa, xhat[:,N:2*N]) + tf.multiply(imag_aa, xhat[:,0:N])
    
    x_out = tf.concat([xout_real,xout_imag],-1)
    return x_out

# 6) sigmoid function:
def sigmoid_fun(x):
    s = 1 / (1 + np.exp(-x))
    return s
    
# 7) save parameters function:
def Save(weight_file):
    dict_name={}
    for varable in tf.trainable_variables():  
        dict_name[varable.name]=varable.eval()   
    sc.savemat(weight_file, dict_name)  


#-----------------------------------------------------------------------------#
# In[4] Define Modules of GEC-SR: Module-A, Module-B, Module-C, Extrinsic operation
    
# Module A (Nonlinear measurements)
def abs_estimation(ob_y,Wvar,pri_z,pri_zvar):
    
    pri_z_abs = tf.sqrt(tf.square(pri_z[:,0:M]) + tf.square(pri_z[:,M:2*M]))
    
    pp_abs = tf.concat([pri_z_abs,pri_z_abs],1)/np.sqrt(2)
    yy_abs = tf.concat([ob_y,ob_y],1)/np.sqrt(2)

    B = tf.divide(2*tf.multiply(pp_abs,yy_abs),(Wvar+pri_zvar))
    
    I0 = tf.minimum( tf.divide(B,(tf.sqrt(tf.square(B)+4))), tf.divide(B,(0.5+tf.sqrt(tf.square(B)+0.25))) )
    
    y_sca = tf.divide(yy_abs,(1+tf.divide(Wvar,pri_zvar)))
    p_sca = tf.divide(pp_abs,(1+tf.divide(pri_zvar,Wvar)))

    zhat = tf.multiply(p_sca + tf.multiply(y_sca,I0), tf.divide(pri_z,pp_abs))
    
    C_constant = tf.divide(pri_zvar, 1 + tf.divide(pri_zvar,Wvar))
 
    zhat_var_c = tf.square(zhat[:,0:M]) + tf.square(zhat[:,M:2*M])
    zhat_var_r = tf.concat([zhat_var_c/2,zhat_var_c/2],1)
    
    zvar = tf.square(y_sca) + tf.square(p_sca) + tf.multiply((1 + tf.multiply(B,I0)), C_constant) - zhat_var_r
    
    return zhat, zvar

# Module Bx (linear trasform Z => AX)
def linear_C2B_each_group(A,mux,varx,muz,varz):
    varx_matrix = tf.matrix_diag(varx)
    varz_matrix = tf.matrix_diag(varz)
    hat_varx = tf.matrix_inverse( varx_matrix +  tf.matmul(tf.matmul(A , varz_matrix,adjoint_a = True), A, adjoint_a = False))
    zz = tf.matmul(varx_matrix,tf.expand_dims(mux,-1), adjoint_a = False) + tf.matmul(tf.matmul(A , varz_matrix,adjoint_a = True), tf.expand_dims(muz,-1), adjoint_a = False)
    hat_mux = tf.matmul(hat_varx, zz, adjoint_a = False)
    hat_mux_out  = tf.squeeze(hat_mux,-1)
    hat_varx_mean_out = tf.matrix_diag_part(hat_varx)
    
    return hat_mux_out, hat_varx_mean_out


# Module Bz (linear trasform AX => Z)
def linear_C2A_each_group(A,mux,varx,muz,varz):

    varx_matrix = tf.matrix_diag(varx)
    varz_matrix = tf.matrix_diag(varz)
    
    hat_varx = tf.matrix_inverse( varx_matrix +  tf.matmul(tf.matmul(A , varz_matrix,adjoint_a = True), A, adjoint_a = False))
    zz = tf.matmul(varx_matrix,tf.expand_dims(mux,-1), adjoint_a = False) + tf.matmul(tf.matmul(A , varz_matrix,adjoint_a = True), tf.expand_dims(muz,-1), adjoint_a = False)
    hat_mux = tf.matmul(hat_varx, zz, adjoint_a = False)
    hat_muz = tf.matmul(A,hat_mux,adjoint_a = False)
    hat_varz = tf.matmul(tf.matmul(A,hat_varx,adjoint_a = False), A, adjoint_b = True)
    hat_muz_out  = tf.squeeze(hat_muz,-1)
    hat_varz_mean_out = tf.matrix_diag_part(hat_varz)
  
    return hat_muz_out, hat_varz_mean_out
    
# Module C (X de-noise)
def Sparse_gaussian_signal(obj_mean,obj_var,obj_rho,rhat,rvar):
    xhat0 = tf.cast(obj_mean,tf.float64)
    xvar0 = tf.cast(obj_var,tf.float64)
    xrho0 = tf.cast(obj_rho,tf.float64)   
   # Compute posterior mean and variance
    a = tf.exp(  tf.divide(-tf.square(rhat),2*rvar) + tf.divide(tf.square(xhat0 - rhat), (2*(xvar0 + rvar))) )
    
   
    c = tf.divide(1, tf.sqrt(tf.multiply(tf.cast(2*math.pi,tf.float64),(rvar + xvar0))))
    
    Z = tf.multiply(tf.divide((1 - xrho0), tf.sqrt( tf.multiply(tf.cast(2*math.pi,tf.float64),rvar))), a ) + tf.multiply(xrho0,c)
    
    xhat_1 = tf.divide(  (xrho0*tf.multiply(c,(tf.multiply(xhat0,rvar) + tf.multiply(rhat,xvar0))))  , (rvar + xvar0) )
    
    xhat = tf.divide( xhat_1,Z)

    x2hat = tf.multiply((xrho0 * tf.divide(c,Z)),   tf.square(  tf.divide(tf.multiply(rhat,xvar0), (rvar + xvar0))) +  tf.divide(tf.multiply(rvar,xvar0),(rvar + xvar0) ) )
    xvar = x2hat - tf.square(xhat)
    
    return xhat, xvar

  
# Extrinsic part: mean & variance
def Ext_part_var(E_new,E_old):
    E_information = tf.divide(E_new, 1- tf.multiply(E_new, E_old))
    return E_information

def Ext_part_mean(E_mean_new,E_var_new,E_mean_old,E_var_old,E_var):
    E_mean_information = tf.multiply(E_var,(tf.divide(E_mean_new,E_var_new) - tf.multiply(E_mean_old,E_var_old)))
    return E_mean_information


    
# In[]: Define train batch:
def Train_batch(sess):
    v_B_= np.zeros([batch_size,]) 
    x_B_= np.zeros([batch_size,]) 
    x_B2C_= np.zeros([batch_size,]) 
    v_B2C_= np.zeros([batch_size,]) 
    z_C2A_= np.zeros([batch_size,]) 
    v_C2A_= np.zeros([batch_size,]) 
    z_A2C_= np.zeros([batch_size,])
    v_A2C_= np.zeros([batch_size,])
    x_C2B_= np.zeros([batch_size,]) 
    v_C2B_ = np.zeros([batch_size,])
    _loss = list()   
    packet=valid_size//train_size
    packet2 = train_size//batch_size
    obj_rho = np.random.uniform(0.5,0.8,(train_size,1))
    
    batch_Y, batch_H, batch_X,x_complex_, batch_un_Y, batch_Z, batch_Xini,batch_D1,batch_SNR = generate_data_iid_test(train_size,M,N,snrdb_train,obj_rho)
    batch_Y = np.squeeze(batch_Y)
    batch_X = np.squeeze(batch_X)
    batch_Z = np.squeeze(batch_Z)
    batch_D = batch_D1/LA.norm(batch_D1,axis=1,keepdims = True)
    

    batch_obj_rho = obj_rho
    batch_obj_SNR = batch_SNR
    
    for offset in range(packet):    
        batch_index_list = (np.random.choice(train_size,train_size,replace=False)).reshape(packet2,batch_size)
        for offset2 in range(packet2):
            batch_index = batch_index_list[offset2]
            batch_Y_b = batch_Y[batch_index]
            batch_H_b = batch_H[batch_index]
            batch_X_b = batch_X[batch_index]
            batch_Z_b = batch_Z[batch_index]
            batch_Xini_b = batch_Xini[batch_index]
            batch_D_b = batch_D[batch_index]
            batch_obj_rho_b =  batch_obj_rho[batch_index]
            batch_obj_SNR_b =  batch_obj_SNR[batch_index]
            
            _, b_loss,x_B_,v_B_,x_B2C_,v_B2C_,z_C2A_,v_C2A_,z_A2C_,v_A2C_,x_C2B_,v_C2B_,v_C_,z_C_,v_C_mean_,x_C_,z_A_,v_A_mean_,damping1_,damping2_,weights_\
            = sess.run([optimizer,cost,X_BB,v_B_mean,x_B2C,v_B2C,z_C2A,v_C2A,z_A2C,v_A2C,x_C2B,v_C2B,hat_varz_mean,hat_muz,hat_varx_mean,hat_mux,z_A,v_A_mean,damping1,damping2,weights],\
                   feed_dict={Y_: batch_Y_b, A_: batch_H_b, X_: batch_X_b, Z_: batch_Z_b, Xini_: batch_Xini_b, D_: batch_D_b, obj_rho_:batch_obj_rho_b, obj_SNR:batch_obj_SNR_b})
        
            print("Packet %d Train loss: %.6f" % ((offset2+1, b_loss)))
            print("damping1:")

            _loss.append(b_loss)
                      
    return _loss, x_B_, v_B_, x_B2C_, v_B2C_, z_C2A_, v_C2A_, z_A2C_, v_A2C_, x_C2B_, v_C2B_,damping1_,damping2_,weights_
    
# In[]: Define train
def Train():
    print("\nTraining ...") 
    saver = tf.train.Saver() 
    with tf.Session() as sess:    
        tf.global_variables_initializer().run()              
#        weight_file=weight_mat+'\deGEC_SR_opt_weight_L1_sampling_1200_exp_A_Epoch_400_maxmin_D_6.mat'

        train_loss_list = []
        for i in range(epochs):      
            train_loss, x_hat_train, v_B_,x_B2C_,v_B2C_,z_C2A_,v_C2A_,z_A2C_,v_A2C_,x_C2B_,v_C2B_,damping1_,damping2_, weights_= Train_batch(sess)
            if math.isnan(np.mean(train_loss)) == 0:
                    weight_file=weight_mat+'\HyperNet_GRU4_atten_%d.mat' %i
                    Save(weight_file)

                    
            
            train_loss_list.append(train_loss)   
        print("\nTraining is finished.")
        saver.save(sess,"./checkpoint_dir/MyModel")
    return train_loss_list,damping1_,damping2_,v_B_,x_B2C_,v_B2C_,z_C2A_,v_C2A_,z_A2C_,v_A2C_,x_C2B_,v_C2B_ ,weights_  
 
# In[]: Define HyperNet

def self_attention_get(Input,Input_Weight_WQ,Input_Weight_WK,d_k_size):
    Input1 = tf.expand_dims(Input,-1)
    Q = Input1*Input_Weight_WQ
    K = Input1*Input_Weight_WK
    d_k = d_k_size
    S = tf.matmul(Q,K,adjoint_b = True)/np.sqrt(d_k)
    attention_weight = tf.nn.softmax(S)
    Output = tf.matmul(attention_weight,Input1)
    Output2 = tf.squeeze(Output)
    return Output2

def damp_GRU(Input_x,Input_h,Input_Weights,d_k_size):
    Inputs = tf.concat([Input_x,Input_h],1)
    r_gate =tf.sigmoid(tf.matmul(Inputs,Input_Weights['W1']))
    z_gate =tf.sigmoid(tf.matmul(Inputs,Input_Weights['W2']))
    h_r = r_gate * Input_h
    Inputs2 = tf.concat([Input_x, h_r],1)
    h_r2 = tf.tanh(tf.matmul(Inputs2, Input_Weights['W3']))
    h_out = (1-z_gate)* Input_h + z_gate * h_r2

    h_out2 = self_attention_get(h_out,Input_Weights['WQ_1'],Input_Weights['WK_1'],d_k_size)
    damping = tf.matmul(h_out2, Input_Weights['W4'])

    return damping, h_out

# In[]: GEC-SR + HyperGRU
with tf.Graph().as_default():
    
    KQ = 10
    
    weights = {
    'W1': tf.Variable(tf.random_normal([2*(N+4), N+4],dtype='float64')/np.sqrt((N+4)/2)),
    'W2': tf.Variable(tf.ones([2*(N+4), N+4],dtype='float64')/np.sqrt((2*(N+4))/2)),
    'W3': tf.Variable(tf.random_normal([2*(N+4), N+4],dtype='float64')/np.sqrt((2*N)/2)),
    'W4': tf.Variable(tf.ones([N+4, 1],dtype='float64')/np.sqrt((N+4)/2)),
    'WQ_1': tf.Variable(tf.ones([1, KQ],dtype='float64')/np.sqrt((KQ)/2)),
    'WK_1': tf.Variable(tf.random_normal([1, KQ],dtype='float64')/np.sqrt((KQ)/2)),
    }
    
    A_ = tf.placeholder(tf.float64,shape=[batch_size,2*M,2*N])
    X_ = tf.placeholder(tf.float64,shape=[batch_size,2*N])
    Y_ = tf.placeholder(tf.float64,shape=[batch_size,M])
    Z_ = tf.placeholder(tf.float64,shape=[batch_size,2*M])
    Xini_ =  tf.placeholder(tf.float64,shape=[batch_size,2*N])  
    D_ =  tf.placeholder(tf.float64,shape=[batch_size,N])
    obj_rho_ = tf.placeholder(tf.float64,shape=[batch_size,1])
    obj_SNR = tf.placeholder(tf.float64,shape=[batch_size,1])
    sigma2 = 10.**(-snrdb_test/10.)/2

   
    D_in = tf.concat([D_,tf.sqrt(obj_SNR)],1)
    Input_hz = tf.zeros((batch_size,N+4), dtype='float64')
    Input_hx = tf.zeros((batch_size,N+4), dtype='float64')
    
    v_B2C = 2*tf.ones((batch_size,2*N), dtype='float64')
    x_B2C = Xini_
    v_A2C = 1/200*tf.ones((batch_size,2*M), dtype='float64')  
    Part_1= tf.matmul(A_,tf.expand_dims(x_B2C,-1))
    z_A2C = tf.squeeze(Part_1,-1)  
    v_C2A = 1/200*tf.ones((batch_size,2*M), dtype='float64')
    z_C2A = z_A2C
    v_C2B = v_B2C
    x_C2B =  x_B2C
    eps=1e-20*tf.ones(1,dtype='float64')
    
    noise_var= sigma2   
    
    for t in range(itermax):      
       
       z_A, v_A = abs_estimation(Y_,noise_var,z_C2A,1/v_C2A) # only for complex  
       v_A_mean = mean_var(v_A)
       
       # Ext
       v_A2C_new = Ext_part_var(v_A_mean,v_C2A)
       z_A2C_new = Ext_part_mean(z_A,v_A_mean,z_C2A,v_C2A,v_A2C_new)
           
       # damping input for GRU      
       if t == 0:
           D_in2 = tf.concat((np.ones([batch_size,2]),v_A2C_new[:,0:1]),1)
           DD_in = tf.concat((D_in,D_in2),1)
           
       if t == 1:
           D_in2 = tf.concat((tf.concat((np.ones([batch_size,1]),damping1),1), v_A2C_new[:,0:1]),1)
           DD_in = tf.concat((D_in,D_in2),1)
           damping1_save = damping1
           
       if t > 1:
           D_in2 = tf.concat((tf.concat((damping1_save,damping1),1), v_A2C_new[:,0:1]),1)
           DD_in = tf.concat((D_in,D_in2),1)
           damping1_save = damping1
           
       damping1, Input_hz = damp_GRU(DD_in,Input_hz,weights, d_k_size = np.float64(KQ))
       
       v_A2C = tf.sigmoid(damping1)*(v_A2C) + (1- tf.sigmoid(damping1)) *(tf.divide(1,v_A2C_new))
       z_A2C = tf.sigmoid(damping1)*(z_A2C) + (1- tf.sigmoid(damping1)) *(z_A2C_new)        
  
       # Linear-mdoel forward (Bx)
       hat_mux , hat_varx_mean = linear_C2B_each_group(A_,x_B2C,v_B2C,z_A2C,v_A2C)
       v_C_mean1 = mean_var(hat_varx_mean)
       # Ext
       v_C2B_new = Ext_part_var(v_C_mean1,v_B2C)
       x_C2B_new = Ext_part_mean(hat_mux,v_C_mean1,x_B2C,v_B2C,v_C2B_new)
       
       v_C2B = (tf.divide(1,v_C2B_new))
       x_C2B = (x_C2B_new) 


       # Module-C
       x_B,v_B = Sparse_gaussian_signal(0,(1/obj_rho_)/2,obj_rho_,x_C2B,1/v_C2B)
       v_B_mean = mean_var(v_B)
          
       # Ext
       v_B2C_new = Ext_part_var(v_B_mean,v_C2B)
       x_B2C_new = Ext_part_mean(x_B,v_B_mean,x_C2B,v_C2B,v_B2C_new)  
       
       # damping input for GRU       
       if t ==0:
           D_in3 = tf.concat((np.ones([batch_size,2]),v_B2C_new[:,0:1]),1)
           DD_in = tf.concat((D_in,D_in3),1)
           
       if t == 1:
           D_in3 = tf.concat((tf.concat((np.ones([batch_size,1]),damping2),1), v_B2C_new[:,0:1]),1)
           DD_in = tf.concat((D_in,D_in3),1)
           damping2_save = damping2
           
       if t > 1:
           D_in3 = tf.concat((tf.concat((damping2_save,damping2),1), v_B2C_new[:,0:1]),1)
           DD_in = tf.concat((D_in,D_in3),1)
           damping2_save = damping2
           
       damping2, Input_hx = damp_GRU(DD_in,Input_hx,weights,d_k_size = np.float64(KQ))
       
       v_B2C = (tf.sigmoid(damping2))*(v_B2C) + (1- tf.sigmoid(damping2))*(tf.divide(1,v_B2C_new))
       x_B2C = (tf.sigmoid(damping2))*(x_B2C) + (1- tf.sigmoid(damping2))*(x_B2C_new)    
        
       #  Linear-mdoel backward (Bz)      
       hat_muz , hat_varz_mean = linear_C2A_each_group(A_,x_B2C,v_B2C,z_A2C,v_A2C)
       v_zC_mean1 = mean_var(hat_varz_mean)
   
       # Ext
       v_C2A_new = Ext_part_var(v_zC_mean1,v_A2C)
       z_C2A_new = Ext_part_mean(hat_muz,v_zC_mean1,z_A2C,v_A2C,v_C2A_new)
       
       # damping
       v_C2A = (tf.divide(1,v_C2A_new))
       z_C2A = (z_C2A_new)  
  
       # Remove disambig phase:
       X_BB = disambig1Drfft(x_B,X_)

       
       # Record the output of different iterative layers to compute loss of model 
       if t == 0:
           xout_iteration = X_BB
           xout_solution  = X_
           
       if t > 0: 
           xout_iteration = tf.concat([xout_iteration,X_BB],0) 
           xout_solution  = tf.concat([xout_solution, X_],0)
          

    cost = tf.nn.l2_loss( xout_iteration -  xout_solution)

    
    learning_rate=5e-2
    
    with tf.variable_scope('opt'):
        optimizer = tf.train.AdamOptimizer().minimize(cost)    
    #Training DetNet
    train_loss_,damping1_,damping2_,v_B_,x_B2C_,v_B2C_,z_C2A_,v_C2A_,z_A2C_,v_A2C_,x_C2B_,v_C2B_, weights_=Train()

    
