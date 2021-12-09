# 236A-LP-Project
## Group 25: Tejas Viswanath, Harish Gondihalli Venkatesha, Jhancy Rani Ramamurthy, Anwesha Chattoraj and Vishwas M Shetty

## Prerequisites:
- Python environment with python version 3.0 or higher
- iPython IDE like Jupyter Labs, Jupyter Notebook, etc
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
- Download all the files in the folder and keep them in the same workspace
- Also download the [MNIST Train Dataset](https://www.kaggle.com/lailaelmahmoudi123/binary-classification-for-the-mnist-dataset/data?select=train.csv) and [MNIST Test Dataset](https://www.kaggle.com/lailaelmahmoudi123/binary-classification-for-the-mnist-dataset/data?select=test.csv) and keep them in the same workspace

## Execution:
## Part 1:
- The `classifier_25.py` is our SVM binary classifier file. Once instantied as an object, it also performs random sample selection and trains itself. The weights and bias are contained in `self.w` and `self.b`.
- The classifier contains 3 main algorithms which are:
  1. Random Percentage sampling:
  2. Epsilon Greedy Sampling:
  3. Error Sampling:
- Run the `main.py` file on your ipython IDE to see the performance of the classifier. You can also see the same performance already plotted in `performance_analysis.ipynb` file.

## Useful Links:
- [Binary Classifer](https://www.kaggle.com/lailaelmahmoudi123/binary-classification-for-the-mnist-dataset)
- [SVM Classifier from scratch](https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2)
- https://link.springer.com/article/10.1007%2Fs13755-017-0023-z
- https://towardsdatascience.com/the-5-sampling-algorithms-every-data-scientist-need-to-know-43c7bc11d17c
- https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/1472-6947-12-8

## References
- https://github.com/sdeepaknarayanan/Machine-Learning/blob/master/Assignment%207/HW7_SVM.ipynb  

- https://www.baeldung.com/cs/svm-hard-margin-vs-soft-margin
- https://www.quora.com/How-do-we-select-which-data-points-act-as-support-vectors-for-SVM
- https://www.cs.cmu.edu/afs/cs/academic/class/15780-s16/www/slides/ml.pdf


## Papers
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



