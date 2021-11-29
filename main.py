from classifier_25 import MyClassifier_25
import pandas as pd

# Load the entire dataset in csv format
dataset = pd.read_csv('mnist/mnist_train.csv')
    
# Load the test data path
test_dataset = pd.read_csv('mnist/mnist_test.csv')

if __name__ == "__main__":
    classfier =  MyClassifier_25(dataset,1,7)
    classfier.selection_and_train()
    print("trained")
    results, performance = classfier.test(test_dataset)
   
    print("Correctly classified ",performance)