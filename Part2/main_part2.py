from classifier_25_Part2 import MyClassifier_25
from GaussianGen import GaussGen 
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'   # To disable the warning messages that we were getting

DatasetToBeUsed="mnist" #"gaussian" or "mnist"

#Numbers to be classified; If using mnist use the appropriate class labels; If using Gaussian use class1=1 and class2=-1
class1=1   #1 if using Gaussian
class2=7  #-1 if using Gaussian


#lambd=[0,0.01,0.1,1.0,10.0,100.0]
lambd=[0.01]

######If using Gaussian Distribution, update the following parameters######

# First set of points with mean=[-1,1] and cov = [[1,0],[0,1]]
mean1 = [-2.0, 2.0]
cov1 = [[1, 0], [0, 1]]  # diagonal covariance

# Second set of points with mean=[1,-1] and cov = [[1,0],[0,1]]
mean2=[2.0,-2.0]
cov2 = [[1, 0], [0, 1]]  # diagonal covariance

# Number of Samples we want to generate and the fraction we want to set aside as test set

# Total number of samples to be generated per mean and cov matric combination. 
# That is here 10000 samples with (mean1, cov1) and another 10000 with (mean2,cov2) are generated.
NoOfSamples = 10000

# Fraction of test in the entire dataset. 
# That is 0.2*(10000+10000) = 4000 test samples.
# We have ensured that 2000 samples are picked from each of the two classes.
FracOfTest = 0.2

########################################


if (DatasetToBeUsed == "mnist"):
    print ("Hey there! You have chosen to run your experiment on digits {} and {} from MNIST dataset.".format(class1,class2)+ "\nWe hope you have updated the paths to the MNIST train and test csv files as per your folder structure")
    #Load the entire dataset in csv format
    dataset = pd.read_csv('mnist/mnist_train.csv')
    
    #Load the test data path
    test_dataset = pd.read_csv('mnist/mnist_test.csv')
else:
    print ("Hey there! You have chosen to run your experiment on Gaussian dataset. \nWe hope you have not set 'class1' to -1 and 'class2' to 1.\nIt has to be 'class1' to 1 and 'class2' as -1")
    gauss = GaussGen(mean1, mean2, cov1, cov2, NoOfSamples, FracOfTest, class1, class2)
    dataset, test_dataset = gauss.DistGen()
    
if __name__ == "__main__":

    for x in lambd:
            classifier =  MyClassifier_25(dataset,class1,class2,x)

            print("Lambda is set to {}".format(x))
            trainlabel_actual,traindata_actual = classifier.prepare_binary(dataset)
            print ("Size of the entire dataset: ", trainlabel_actual.shape[0])
            classifier.train(traindata_actual,trainlabel_actual)
            results, performance = classifier.test(test_dataset)
            correct = classifier.assess_classifier_performance(performance)
            print("Correctly classified when entire dataset is used: ",correct)

            selected_Df_logic1 = classifier.ILP(dataset)
            trainlabel_logic1,traindata_logic1 = classifier.prepare_binary(selected_Df_logic1)
            print ("Size of the dataset after Logic 1: ", trainlabel_logic1.shape[0])
            classifier.train(traindata_logic1,trainlabel_logic1)    
            results, performance = classifier.test(test_dataset)
            print("Trained using logic 1 dataset")
            correct = classifier.assess_classifier_performance(performance)
            print("Correctly classified using ILP reduced train set: ",correct)

            print ("Now let us run K-NN on the ILP reduced dataset")
            reduced_logic1 = classifier.dis_sim_ngbr(selected_Df_logic1)
            trainlabel_logic1_KNN,traindata_logic1_KNN = classifier.prepare_binary(reduced_logic1)
            print ("Sample size after KNN: ", trainlabel_logic1_KNN.shape[0])
            classifier.train(traindata_logic1_KNN,trainlabel_logic1_KNN)
            results, performance = classifier.test(test_dataset)
            print("Trained using KNN on ILP reduced train dataset")
            correct = classifier.assess_classifier_performance(performance)
            print("Correctly classified using KNN on ILP reduced set: ",correct)
