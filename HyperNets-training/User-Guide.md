# Training GEC-SR Hypernetworks for "Phase Retrieval"
(c) 2021 Chang-Jen Wang and Chao-Kai Wen e-mail: dkman0988@gmail.com and chaokai.wen@mail.nsysu.edu.tw

--------------------------------------------------------------------------------------------------------------------------
# Information:
- GEC-SR-HyperNet: Generalized expectation consistent signal recovery based on HyperNet with Attention
- GEC-SR-HyperGRU: Generalized expectation consistent signal recovery based on dynamic HyperNet with Attention

For phase retrieval, GEC-SR based on sutiable damping factors to get good performance. However, the learning parameters of the existing unfolded algorithms are trained for a specific task of image recovery. Retraining the parameters is often needed in a clinical setting, where different forward models (e.g., measurement distribution and size, and noise level) may be used; otherwise, the stability and optimality of the learned algorithm will be lost. Instead of learning a set of optimal damping factors directly, the hypernetwork learns how to generate the optimal damping factors according to the clinical settings, thereby ensuring its adaptivity to different scenarios. For details, please refer to 

C. J. Wang, C. K. Wen, S. H. Tsai, S. Jin, and G. Y. Li, Phase Retrieval using Expectation Consistent Signal Recovery Algorithm based on Hypernetwork, IEEE Trans. signal process., 2021, to appear.

We provide the training codes in a way that you can perform based on re-training for different related channel of phase retrieval.


# How to start a training:
- Step 1. Install python (3.5.2) and tensorflow (v1.14)

- Step 2. Add folders (i.e., X_ini2.py & AA.py & AB.py) to the same directory
  
- Step 3. Now, you are ready to run the training code:
- AA.py train GEC-SR-HyperNet
- AB.py train GEC-SR-HyperGRU

  main_phase_retrieval
  


