#INSTRUCTIONS for EX. on SVC parallelization

After modifying the class  svm.cpp file with OPENMP optimization
and after setting in the svm.cpp the number of threads with the
command 
omp_set_num_threads(4);

we can install the modified version of the scikit library in 
our conda environment (called sklearn-env) launching the following command

pip install -e . -v
from the main directory of the scikit repo. 

After installing it we can use the python program 'Testing_SVC_on_IRIS.py' (at the path Day6/Kernel_methods_for_ML/classification/ inside the school repo), and in particular the function calls at line 18 and 33, to time the speedup obtained with the parallelization.


We can use the original version of scikit-learn in the environment /leonardo_work/ICT23_SMR3872/shared-env/Gabenv)
on the Leonardo cluster, to measure the runtime for the serial version of SVC (optional: you can measure also SVR runtime).
