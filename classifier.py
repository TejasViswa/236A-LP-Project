import numpy as np

class myclassifier:
    def __init__(self) -> None:
        pass

    def sample_selection(self,training_sample,):
        """
        training_sample = row of dataset that is input to be determined whther or nt it will be added to the dataset
        Return:
            Output of this function should be the MyClassifier object that keeps the updated data set
        """
        pass

    def train(self,train_data,train_label):
        """
        train_data : the rows of the training data set. N is the number of rows, and M is the number of features 
        train_label : the labebls of the rows 

        Return:
            Output of this function should be the MyClassifier object which has the information of weights, bias, number of classes, number of features
        """
        pass

    def f(self,input):
        """
        Returns:
            Output is an estimated class
        """
        np.dot(input,self.W) + self.w

    def test(self):
        pass