import numpy as np
import pandas as pd

class MyClassifier_25:
    def __init__(self,dataset = np.zeros((100,785))):
        
        # By default, a 100 x 785 zero matrix is taken as dataset
        if not isinstance(dataset, pd.DataFrame):
            dataset = np.zeros((100,785))
            print("Dataset is not a Pandas DataFrame")
        
        # Separating training data and training label
        self.train_data, self.train_label = dataset.drop(labels = ["label"],axis = 1).to_numpy(), dataset["label"]
        
        # W is matrix of m rows and L columns and w is a L - dimensional vector
        # Assuming L is 1, which makes W as (m features x 1) vector and w as scalar:
        self.W = np.zeros((self.train_data.shape[1],1))
        self.w = np.zeros(1)


    def sample_selection(self,training_sample):
        """
        training_sample = row of dataset that is input to be determined whther or nt it will be added to the dataset
        Return:
            Output of this function should be the MyClassifier object that keeps the updated data set
        """
        
#         if not isinstance(x, pd.DataFrame):
#             Sample = np.zeros((1,785))
#             print("Sample is not a Pandas DataFrame")

        # Separate Features and Labels
        sample_X,sample_Y = training_sample[:-1], training_sample[-1]
        
        # Some logic to check if sample data is useful
        logic = True
        
        # Updating training dataset after verifying some logic
        if(logic):
            self.train_data = np.vstack([self.train_data,sample_X])
            self.train_label = np.vstack([self.train_label,sample_X])
        
        return self

    def train(self,train_data,train_label):
        """
        train_data : the rows of the training data set. N is the number of rows, and M is the number of features 
        train_label : the labebls of the rows 

        Return:
            Output of this function should be the MyClassifier object which has the information of weights, bias, number of classes, number of features
        """
        
        # calculate hinge loss
        N = train_data.shape[0]
        distances = 1 - self.f(train_label) * np.matrix.dot(self.W.getT(),train_label) + self.w
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = reg_strength * (np.sum(distances) / N)

        # calculate cost
        cost = 1 / 2 * np.dot(self.W, self.W) + hinge_loss
        
        return self

    def f(self,input):
        """
        input = this is the input vector to f(.), which corresponds to the function g(y) = W'*y + w which is an L dimensional vector
        Return:
            Output is an estimated class
        """
        # Let us first define a variable for the function g(y) = W'*y + w
        g = np.matrix.dot(self.W.getT(),input) + self.w
        
        # We return estimated class as +1 or -1 when g is positive and non-positive respectively
        s = 1 if all([x > 1 for x in g]) else -1
        
        return s
        
    def test(self):
        pass
