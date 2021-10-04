# GEC-SR-PR-HyperNets
Phase Retrieval using Expectation Consistent Signal Recovery Algorithm based on Hypernetwork

(c) 2021 Chang-Jen Wang and Chao-Kai Wen e-mail: dkman0988@gmail.com and chaokai.wen@mail.nsysu.edu.tw

--------------------------------------------------------------------------------------------------------------------------
# Information:
- GEC-SR-HyperNet: Generalized expectation consistent signal recovery based on HyperNet with Attention
- GEC-SR-HyperGRU: Generalized expectation consistent signal recovery based on dynamic HyperNet with Attention

For phase retrieval, GEC-SR based on sutiable damping factors to get good performance.
However, the learning parameters of the existing unfolded algorithms are trained for a specific task of image recovery. 
Retraining the parameters is often needed in a clinical setting, where different forward models (e.g., measurement distribution and size, and noise level) may be used; otherwise, the stability and optimality of the learned algorithm will be lost.
Instead of learning a set of optimal damping factors directly, the hypernetwork learns how to generate the optimal damping factors according to the clinical settings, thereby ensuring its adaptivity to different scenarios. For details, please refer to 

C. J. Wang, C. K. Wen, S. H. Tsai, S. Jin, and G. Y. Li, Phase Retrieval using Expectation Consistent Signal Recovery Algorithm based on Hypernetwork, IEEE Trans. Signal Process. accepted in Oct. 2021.

Here, we provide the training codes in a way that you can perform based on re-training for different related channel of phase retrieval, and the testing codes compare HyperNets with prior algorithms

# Training (Python) and Testing (Matlab) codes:
- Training (Python): HyperNets-training file 
- Testing (Matlab): HyperNets-test file

  
