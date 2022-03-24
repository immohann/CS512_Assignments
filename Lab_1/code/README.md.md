# CS512_Project

Overview:

In this Lab, We have implemented a conditional random field for optical character recognition (OCR), with emphasis on inference and performance test.

The Lab consists of 5 parts listed below:

1) CRF : We have implemented a decoder with computational cost of O(m|Y|^2) that is based on the max-sum algorithm.
         
2) Training CRF : We have implemented a Dynamic Programming Algorith to compute log(p(y|X)) and its gradient with respect to Wj and Tij, based on marginal distribution of Yj and (Ys,Ys+1). We have also cross-verified the correctness of our gradients using scipy.optimize.check_grad(). Furthermore, We have optimized the values of Wj and Tij using fmin_tnc from scipy.optimize. The decoder from part 1 has then been used on the test data to measure it's performance.

3)  Benchmarking with Other Learning Models: We have compared the performance of our CRF decoder with SVM-MC and SVM-Struct using LinearSVC and off the shelf SVMHMM packages. The performances for all the 3 models have been compared based on 2 parameters: 
i) Letter-Wise prediction accuracy and ii)Word-Wise prediction accuracy on the test data.

4) Stochastic Optimization : We have compared the efficiencies of  SGD and momentum based stochastic optimization methods agains LBFGS for reducing the test data error. Word-wise test error and Training objective value has been used for this comparison. Furthermore, We have implemeneted MCMC sampler for sampling gradients for various number of samples limits and chosen the one that works most efficiently. These gradients are again passed to SGD, momentum and LBFGS models and their efficiencies have been compared. Finally, Rao-Blackwellization approach has been used to get better node and edge distributions as well as reduce the variance. The values of node and edge marginals before and after Rao-Blackwellizations for the DP model have then been compared.

5)  Robustness to Distortion : We have now investegated a scenario where the training data images have been rotatied and translated. The predection acuracy of thev CRF decoder has been evaluated on this distorted data and compared with the performance of SVM-MC algorithm on the distorted images. Same 2 parameters have been used for performance evaluation as in part 3.


Running the code:

all_functions.py and read_data.py contain all the helper functions for different parts of the lab.

1) 1c has been implemented in 1c.py and the result for the same has been provided in the report.

2) Most of the base functions for 2 and the following questions have been written in all_functions.py. To run 2, ref_optimize.py needs to run. You can run the ref_optimize function to get the optimal values of W and T. The optimal values can be found in Solution.txt

3) SVM.py --- The optimal parameters have been hypertuned based on different C values and can be found under result/c_equal_[C].txt where C=1, 10, 100, 1000. The plots are present under /plots/Q3 and have been included in the report as well.

4) For question 4a: You can run 4a_SGD.py and 4a_SGD_with_momentum.py respectively. 
For 4b, the implementation can be found in mcmc_sampling.py.


5) evil_machine_learner.py -- The optimal parameters have been hypertuned based on different C values under result/solution[C]distortion.txt where C=500, 1000, 1500, 2000. The plots are present under /plots/Q4 and have been included in the report as well.

All the plots can be found in our report.
