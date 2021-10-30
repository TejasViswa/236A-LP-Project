import numpy as np

class myclassifier:
    def __init__(self):
        self.W = np.zeros()
        self.w = np.zeros()
        self.train_data = np.zeros()
        self.train_label = np.zeros()

    def sample_selection(self,training_sample,):
        """
        training_sample = row of dataset that is input to be determined whther or nt it will be added to the dataset
        Return:
            Output of this function should be the MyClassifier object that keeps the updated data set
        """
        
        # Separate Features and Labels
        sample_X,sample_Y = training_sample[:-2], training_sample[-1]
        
        # Some logic to check if sample data is useful
        
        # Updating training dataset
        self.train_data.append(sample_X)
        self.train_label.append(sample_Y)
        
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
        distances = 1 - f(train_label) * np.matrix.dot(self.W.getT(),train_label) + self.w
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
        
        # We return estimated class as +1 or -1 when g is non-negative and negative respectively
        return 1 if g >= 0 else -1
        
    def test(self):
        pass
