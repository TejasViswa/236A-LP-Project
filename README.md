# 236A-LP-Project
## Group 25: Tejas Viswanath, Harish Gondihalli Venkatesha, Jhancy Rani Ramamurthy, Anwesha Chattoraj and Vishwas M Shetty

## Report:
https://www.overleaf.com/read/nphhyqkkyyps

## Prerequisites:
- Python environment with python version 3.0 or higher
- (Optional) iPython IDE like Jupyter Labs, Jupyter Notebook, etc
- Numpy Module (Run these exact commands on the command prompt of the specific environment[eg:anaconda terminal if applicable])
  - `pip install numpy`
- pandas Module
  - `pip install pandas`
- matplotlib Module
  - `pip install matplotlib`
- seaborn Module
  - `pip install seaborn`
- cvxpy Module
  - `pip install cvxpy`
- Download all the files in the folder and keep them in the same workspace.
- Also download the [MNIST Train Dataset](https://www.kaggle.com/lailaelmahmoudi123/binary-classification-for-the-mnist-dataset/data?select=train.csv) and [MNIST Test Dataset](https://www.kaggle.com/lailaelmahmoudi123/binary-classification-for-the-mnist-dataset/data?select=test.csv) and keep them in the same workspace as the `classifier.py`.

## Execution:
## Part 1:
- Inside `Part1` folder, the `classifier_25.py` is our SVM binary classifier file. Once instantied as an object, it also performs random sample selection and trains itself. The weights and bias are contained in `self.w` and `self.b`.
- The classifier contains 3 main algorithms which are:
  1. Random Percentage sampling: Random sampling of data based on perentage
  2. Epsilon Greedy Sampling: Greedy Sampling based on region and prediction
  3. Error Sampling: Sampling based on incorrect predictions
- To instantiate an object of classifier class, run it as `my_clf = classifier_25(dataset,class1,class2,algorithm)`,
  where:
  - `dataset` is the input data dataset
  - `class1` and `class2` are the labels for the two classes. (eg: 1 and 7, 4 and 9 for MNIST and 1 and -1 for synthetic)
  -  `algorithm` is the algorithm number which can be 1, 2 and 3 based on desired algorithm as described in previous point.
- Run the `main.py` file inside `Part1` folder on your ipython IDE to see the performance of the classifier. You can also see the same performance already plotted in `performance_analysis.ipynb` file.

## Part 2:

- Please note that we have implemented the ILP using numpy, instead of using cvxpy.
- Inside `Part2` folder, we have:
  - `classifier_Part2.py`: Implementation of all the functions. The function ILP() in this script is our ILP implementation. As stated before, we have not used cvxpy - formulate this LP problem, instead we have used numpy.
  - `GaussianGen.py`: Script to generate the synthetic dataset samples.
  - `main_part2.py`: The top level script to be run.
- Following are the parameters to be set in `main_part2.py`:
  1. `DatasetToBeUsed`: `mnist` or `gaussian` 
  2. `class1` and `class2`: Labels that you want to classify. In case of Gaussian, please set `class1` to 1 and `class2` to -1.
  3. `lambd`: Value of the regulariation parameter.
  4. `mean1`, `mean2`, `cov1`, and `cov2`: Mean and Covariance matrix values. This is if using Gaussian dataset.
  5. `NoOfSamples`: Number of samples to be generated per (mean1,cov1) and (mean2, cov2) combination. This is if using Gaussian dataset.
  6. `FracOfTest`: Fraction of (mean1,cov1) and (mean2, cov2) to be set aside as test set. This is if using Gaussian dataset.
  7. `dataset` and `test_dataset`: Update paths as per your folder structure.
- To execute part 2, run the file - `python main_part2.py`

## notes on the algorithms used 

Epsilon-greedy algorithm 
### Choosing support vectors 
The goal  of the project is to choose the data points coming in from the peripheral nodes most efficiently. This involves choosing the points that would change the hyperplane the most- these would be the support vectors for the current hyperplane separating the points. 

### Approaching the problem 
### Using Clusters


### Algorithm 1 - Epsilon greedy sampling 
Send a certain number of samples at the beginning to get an initial hyperplane. After the initial batch of points are sent uncritically, the peripheral nodes use this initial classifier to classify every incoming point. The classification output is either -1 or 1 if the hyperplane is able to classify the point into a separable class, the point lies in the p1 or p2 region of this hyperplane (outside the margin area) - it is not a support vector and it reinforces the current hyperplane. It is sent to the central node with a probability $\epsilon_{out}$.  If the classifier incorrectly classifies the point (say -1 as 1) the point, but the point is not in the margin region of the SVM it is sent to the central node with a probability $1-\epsilon_{out}$ . If the output of the classifier function is not -1 or 1 i.e it was not able to classify the data point into one of the 2 classes, it is lies between the two margins of the SVM, a support vector [sic], it influences the hyperplane formation and is sent to the central node with a probability $\epsilon_{sv}$.

This algorithm is based on the simple understanding that the the support vectors influence the hyperplane more, and that incorrectly classified points should also be accounted for in the hyperplane formation and sent to the central node so we can improve it. 

But we need to do better. 

Check 2.4.4 Stochastic Gradient Descent for the Soft-SVM Problem in this link
https://scholarscompass.vcu.edu/cgi/viewcontent.cgi?article=5368&context=etd
### 
## Credits:
- [Selecting training sets for support vector machines: a review](https://link.springer.com/content/pdf/10.1007/s10462-017-9611-1.pdf)
- [Clustering Model Selection for Reduced Support Vector Machines?](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.102.9904&rep=rep1&type=pdf)
- [Support vector machines based on K-means clustering for real-time business intelligence systems](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.96.7836&rep=rep1&type=pdf)
- [Selection of Candidate Support Vectors in incremental SVM for network intrusion detection](https://www.sciencedirect.com/science/article/pii/S0167404814000996)
- [Online support vector machine based on convex hull vertices selection](https://pubmed.ncbi.nlm.nih.gov/24808380/)
- [The Huller: A Simple and Efficient Online SVM](https://link.springer.com/content/pdf/10.1007/11564096_48.pdf)    
- [ Online training of support vector classifier](https://www.sciencedirect.com/science/article/pii/S0031320303000384)
- [Active Learning with Support Vector Machines](http://image.diku.dk/jank/papers/WIREs2014.pdf)
- [SVM-Based Spam Filter with Active and Online Learning](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.109.5961&rep=rep1&type=pdf)
- [Reducing the Number of Training Samples for Fast Support Vector Machine Classification](https://www.researchgate.net/profile/Saman-Halgamuge/publication/235009071_Reducing_the_number_of_training_samples_for_Fast_Support_Vector_Machine_Classification/links/0f31752ee3e62e4790000000/Reducing-the-number-of-training-samples-for-Fast-Support-Vector-Machine-Classification.pdf)
- [SVM Presentation](https://edisciplinas.usp.br/pluginfile.php/5078086/course/section/5978682/svm2.pdf)
- [Online Learning: A Comprehensive Survey](https://arxiv.org/pdf/1802.02871.pdf)
- [Reduction of Training Data Using Parallel Hyperplane for Support Vector Machine](https://www.tandfonline.com/doi/full/10.1080/08839514.2019.1583449?scroll=top&needAccess=true)
- [Binary Classifer](https://www.kaggle.com/lailaelmahmoudi123/binary-classification-for-the-mnist-dataset)
- [SVM Classifier from scratch](https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2)
- https://link.springer.com/article/10.1007%2Fs13755-017-0023-z
- https://towardsdatascience.com/the-5-sampling-algorithms-every-data-scientist-need-to-know-43c7bc11d17c
- https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/1472-6947-12-8
- https://github.com/sdeepaknarayanan/Machine-Learning/blob/master/Assignment%207/HW7_SVM.ipynb  
- https://www.baeldung.com/cs/svm-hard-margin-vs-soft-margin
- https://www.quora.com/How-do-we-select-which-data-points-act-as-support-vectors-for-SVM
- https://www.cs.cmu.edu/afs/cs/academic/class/15780-s16/www/slides/ml.pdf
