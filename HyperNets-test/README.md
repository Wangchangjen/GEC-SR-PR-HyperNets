# Simulation for "Decentralized Expectation Consistent Signal Recovery for Phase Retrieval"
(c) 2020 Chang-Jen Wang and Chao-Kai Wen e-mail: dkman0988@gmail.com and chaokai.wen@mail.nsysu.edu.tw

--------------------------------------------------------------------------------------------------------------------------
# Information:
- GEC-SR: Generalized expectation consistent signal recovery
- deGEC-SR: Decentralized expectation consistent signal recovery

For phase retrieval, GEC-SR is good performance and deGEC-SR is good efficient algorithm. For details, please refer to 

C. J. Wang, C. K. Wen, S. H. Tsai, and S. Jin, Decentralized Expectation Consistent Signal Recovery for Phase Retrieval, IEEE Trans. signal process., 2020, to appear.

We provide the codes in a way that you can perform based on the simulator for "Phase retrieval". Therefore, you can compare several different phase retrieval algorithms under the same setting.


# How to start a simulation:
- For phase retrieval, the prior algorithms (e.g., Fienup, Wirtflow, Gerchbergsaxton, Phasemax, PhaseLamp, etal.) can download from
  https://github.com/tomgoldstein/phasepack-matlab

- Step 1. Download our proposed phase retrieval algorithms (GEC_SR.m & DeGEC_SR.m), which can be found

  https://github.com/Wangchangjen/Matlab_deGEC-SR
  
- Step 2. Add folders (i.e., Estimation, Operator, Solvers) to the executive directory
  
- Step 3. In main_phase_retrieval.m, find the line 

  You can select GEC-SR or deGEC-SR based on L. L=1 is GEC-SR and other algorithms. L > 1 is deGEC-SR.
  
- Step 4. Now, you are ready to run the phase retrieval:

  main_phase_retrieval
  
--------------------------------------------------------------------------------------------------------------------------------------
The simulator returns a plot of the MSE of iterations for our proposed phase retrieval algorithms.
<div align=center><img width="600" height="600" src="https://github.com/Wangchangjen/Matlab_deGEC-SR/blob/master/EXAMPLE.png"/></div>

