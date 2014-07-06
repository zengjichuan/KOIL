Kernelized Online Imbalanced Learning with Fixed Buddget
Implemented by Junjie Hu
Contact: jjhu@cse.cuhk.edu.hk

Brief: This is the C++ version KOIL. This program depends on the boost library, which is popular with std C++ users.
For boost installation, please refer to http://www.boost.org/.

This program is tested on Ubuntu 12.04 and 13.10. Open the terminal and cd to the folder. Then type as follow:
(1) cmake .
(2) make

Three executable functions are generated.
1. online evaluation:
Format: ./online <dataset_file> <C> <gamma> <loss type>
E.g.: ./online diabetes 1 0.25 l2
Perform 20-runs KOIL based l2 loss, with C=1 and gamma=0.25.

2. CV: cross validation by exploniential step
Format: ./CV <dataset_file> <Num of clist> <Num of glist> <c start> <g start> <cstep><gstep> <cvfold> <loss type>
E.g.: ./CV diabetes 10 20 -5 -10 2 2 5 l1
Perform 5-fold cross validation of KOIL based on l1 loss on diabetes dataset, where C in range [2^(-5:1:4)], gamma in range [2^(-10:1:9)].
 
3. CVP: cross validation by additional step
Format: ./CVP <dataset_file> <Num of clist> <Num of glist> <c start> <g start> <cstep><gstep> <cvfold> <loss type>
E.g.: ./CVP diabetes 10 20 1 0.001 10 0.05 5 l1
Perform 5-fold cross validation of KOIL based on l1 loss on diabetes dataset, where C in range [1:10:91], gamma in range [0.01:0.05:0.901].
 
