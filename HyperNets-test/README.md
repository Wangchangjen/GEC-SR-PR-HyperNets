# Testing with Matlab for HyperNets of GEC-SR
(c) 2021 Chang-Jen Wang and Chao-Kai Wen e-mail: dkman0988@gmail.com and chaokai.wen@mail.nsysu.edu.tw

--------------------------------------------------------------------------------------------------------------------------
# How to start a simulation:
- For phase retrieval, the prior algorithms (e.g., TAF, RWF, HIO, Fienup, Wirtflow, Gerchbergsaxton, Phasemax, PhaseLamp, etal.) can download from
  https://github.com/tomgoldstein/phasepack-matlab

- Step 1. Download our proposed phase retrieval algorithms (i.e., "Matlab_code" file).
  
- Step 2. Add folders (i.e., Matlab_code) to the executive directory
  
- Step 3. main_FFTBased_DE.m run the testing simulator.<br>
GECSR_Hyper.m is PR with GEC-SR HyperNet<br>
GECSR_GRU.m is PR with GEC-SR HyperGRU<br>
PR_algorithms.m is PR with prior algorithms<br>
Note:  HyperNet_NN_mul4_attention.mat and HyperNet_GRU_attention.mat are the trained weights for HyperNet and HyperGRU, respectively.

--------------------------------------------------------------------------------------------------------------------------------------
The simulator returns a plot of the MSE of iterations for our proposed algorithms and convergence performance of prior algorithms.
<div align=center><img width="600" height="600" src="https://github.com/Wangchangjen/GEC-SR-PR-HyperNets/blob/main/HyperNets-test/Matlab_code/Result.png"/></div>

