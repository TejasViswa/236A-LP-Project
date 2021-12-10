from classifier_25 import MyClassifier_25
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Load the entire dataset in csv format
# dataset = pd.read_csv('mnist/mnist_train.csv')
mnist_train = pd.read_csv("Part1/mnist/mnist_test.csv")

# Load the test data path
# test_dataset = pd.read_csv('mnist/mnist_test.csv')
mnist_test = pd.read_csv("Part1/mnist/mnist_test.csv")

#Take copies of the master dataframes
train = mnist_train.copy()
tester = mnist_test.copy()

# Change Classes to Test here MNIST
class1 = 1
class2 = 7

tester = tester.loc[tester['label'].isin([class1,class2])]

train_class_1 = train.loc[train['label'] == class1]
train_class_2 = train.loc[train['label'] == class2]
train = train.loc[train['label'].isin([class1,class2])]
train_rdm = train.sample(frac = 0.5)
test_rdm = train.sample(frac = 0.2)


def average_testing_across_multiple_classifiers(num = 5, min_len= 10):
    # Testing the Algorithm performance by averaging from 10 classifiers
    num = 5
    min_len = 100
    # Outer Loop for algorithm
    for i in range(1,3):
        # Inner Loop for averaging accuracy performance
        avg = None
        for j in range (num):
            my_clf = MyClassifier_25(train,4,9,i)
            x,y = my_clf.plot_classifier_performance_vs_number_of_samples(tester,False)
            if avg is None:
                avg = y
            else:
                avg = np.add(np.array(avg),np.array(y)).tolist()
        avg = (np.array(avg) / num).tolist()
        if min_len > len(avg):
            min_len = len(avg)
        if min_len < len(avg):
            avg = avg[0:min_len]
            x = range(my_clf.batch_size,(min_len+1)*my_clf.batch_size,my_clf.batch_size)
        plt.plot(x,avg)
        plt.title("Algorithm No.: %i"%i)
        plt.xlabel("No. of Samples")
        plt.ylabel("Accuracy")
        plt.show()
        print("Avg:",avg)

if __name__ == "__main__":
    classfier =  MyClassifier_25(train,class1,class2)
    print("trained")
    results, performance = classfier.test(test_rdm)
    print("Correctly classified ",performance)
    x,y = classfier.plot_classifier_performance_vs_number_of_samples(tester)
    
