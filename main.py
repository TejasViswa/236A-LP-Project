from classifier_25 import MyClassifier_25
import pandas as pd

# Load the entire dataset in csv format
dataset = pd.read_csv('/Path/to/train/mnist_train.csv')
    
# Load the test data path
test_dataset = pd.read_csv('/Path/to/test/mnist_test.csv')

if __name__ == "__main__":
    classfier =  MyClassifier_25()