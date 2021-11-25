import numpy as np
import pandas as pd
import cvxpy as cp
import random

from pandas.io.pytables import performance_doc



class MyClassifier_25:  

    def __init__(self,dataset,class1:int,class2:int) -> None:
        self.w = None
        self.b = None
        self.classes = { 1 : class1, -1: class2, 0:None}
        self.dataset_train = dataset

        #data prep
        self.trainlabel,self.traindata = self.prepare_binary(self.dataset_train)
        self.yet_to_train_dataset = self.dataset_train
        self.sampled_dataset = None
        
        self.i = 0 # Dataset Iterator 
        self.sel_arr = np.zeros(self.traindata.shape[0]) # Binary Array indicating whether a sample -
                     # is selected or not in order of sample selection and not dataset index
        
        #Change these variables:
        self.perct_sel_smpls = 0.3 # percentage of Selected samples from dataset DEFAULT VALUE
        self.batch_size = 100 # Batch Size for samples
        self.mini_batch_size = 20 # Mini Batch Size for samples
        self.mini_batch_slots_to_be_filled = int(self.perct_sel_smpls * self.mini_batch_size)
        self.batch_slots_to_be_filled = int(self.perct_sel_smpls * self.batch_size)

        self.epsilon = 0.5
        self.exploit_perc = 0.8 # Percentage of samples that are close to svm
        self.explore_perc = 1 - self.exploit_perc # Percentage of samples that are random
        # UNDERSTAND THAT EXPLORE AND EXPLOIT PERCENTAGES ARE PERCENTAGES FROM PERCENTAGE OF SELECTED SAMPLES
        # ie: if perct_sel_smpls = 0.3 and exploit_perc = 0.8,
        # then exploit_perc is essentially 0.8x0.3 = 0.24 of original dataset
        
        #train the classfier 
        # self.selection_and_train()

        
    
    def prepare_binary(self,dataset):
        #USAGE    
        # Since we have to deal with a binary classifier to diffrentiate between digits 7 and 1, 
        # we choose only those examples.
        # If asked to train a classifier on any other pair a, b (say),
        # please pass the right arguments to the following function as follows:
        # trainlabel, traindata, dataTargetDf = prepare_binary(a,b)


        # We now assign +1 to one class and -1 to the other;
        # Therefore +1 and -1 will be the new labels
        class1 = self.classes[1]
        class2 = self.classes[-1]

        trainlabel = dataset.loc[(dataset['label']== class1)  | (dataset['label']== class2) ]['label']
        trainlabel.loc[trainlabel == class1] = 1
        trainlabel.loc[trainlabel == class2] = -1
        trainlabel = trainlabel.to_numpy()
    
        #In order to match dimensions of "traindata" and "trainlabel", we convert trainlabel to two dimension array
        # for hinge loss
        trainlabel= np.reshape(trainlabel, (trainlabel.shape[0],1))   

        # We now extract the features for the two classes
        traindata = dataset.loc[(dataset['label']== class1)  | (dataset['label']== class2) ]
        traindata = traindata.drop(labels = ["label"],axis = 1)
        traindata = traindata.to_numpy()

        # print(traindata.shape[1])



        return trainlabel, traindata

    def target_df(self,traindata,trainlabel):
        # Also creating a dataframe with these, so that we can randomize the order of the train data when needed without
        # losing the mapping between feature vectors and the target labels
        trainDf=pd.DataFrame(traindata)
        targetDf=pd.DataFrame(trainlabel,columns=['target'])
        
        dataTargetDf = pd.concat([trainDf, targetDf[['target']]], axis = 1)
        ##If randomizing the order, should we use the dataframe 'finalDf'?
        return dataTargetDf

    def subset(self,dataTargetDf, subsetfrac:float):
        
        # Usage: If 20% of the data is to be randomly selected
        # subsetDf = subset(dataTargetDf, 0.2)
        
        return dataTargetDf.sample(frac = subsetfrac)
    
    def sample_selection(self,training_sample):
        
        # This method accepts only 1 random training sample at a time and decides whether to send it or not
        
        # ==MADE CLASS VARIABLES, REVERT IF SOMETHING BREAKS==
        # mini_batch_slots_to_be_filled = int(self.perct_sel_smpls * self.mini_batch_size)
        # batch_slots_to_be_filled = int(self.perct_sel_smpls * self.batch_size)

        mini_start = (self.i // self.mini_batch_size)*self.mini_batch_size
        mini_end = mini_start + (self.i % self.mini_batch_size) + 1

        print("mini_start: ",mini_start,"mini_end: ", mini_end)
        print("Mini Batch: ",self.sel_arr[mini_start:mini_end])
        
        # EXPLORE =====================================================================
        # Random Binary value is assigned initially
        accept_sample = random.randint(0, 1)
        
        lbl, dt = self.prepare_binary(training_sample)

        # EXPLOIT =====================================================================
        # after 50 percent completion, choose p1 and p3 over p2
        if ((self.i/self.dataset_train.shape[0])>=0.5 and (self.i/self.dataset_train.shape[0])<0.75 and \
            np.count_nonzero(self.sel_arr[mini_start:mini_end]) < self.mini_batch_slots_to_be_filled*self.exploit_perc):
            if(self.region(dt)=='p1' or self.region(dt)=='p3'):
                accept_sample = 1
        # after 75 percent completion, choose p2 over p1 and p3
        elif ((self.i/self.dataset_train.shape[0])>=0.75 and \
             np.count_nonzero(self.sel_arr[mini_start:mini_end]) < self.mini_batch_slots_to_be_filled*self.exploit_perc):
            if(self.region(dt)=='p3'):
                accept_sample = 1
        
        # MINI BATCH CREATION =================================================
        # No. of mini batch and batch slots that must be filled to satisfy percentage criteria
        mini_batch_count = np.count_nonzero(self.sel_arr[mini_start:mini_end])
        if mini_batch_count >= self.mini_batch_slots_to_be_filled:
            accept_sample = 0
        
        print("If Mini Batch Count: ",mini_batch_count," >= mini slots ",self.mini_batch_slots_to_be_filled," then 0")
            
        # Lower bound for mini batch percentage criteria
        if (self.i % self.mini_batch_size) >= (self.mini_batch_size - self.mini_batch_slots_to_be_filled) and mini_batch_count <self.mini_batch_slots_to_be_filled:
            accept_sample = 1
        
        print("If Mini Batch iterator: ",self.i % self.mini_batch_size," >= rem mini slots ",self.mini_batch_size - self.mini_batch_slots_to_be_filled, "and Mini Batch Count: ",mini_batch_count," < mini slots ",self.mini_batch_slots_to_be_filled," then 1")

        ### BATCH SIZE INFORMATION HAS TO COME FROM CENTRAL NODE
        start = (self.i // self.batch_size)*self.batch_size
        end = start + (self.i % self.batch_size) + 1
        batch_count = np.count_nonzero(self.sel_arr[start:end])
        if batch_count >= self.batch_slots_to_be_filled:
            accept_sample = 0
        
        print("start: ",start,"end: ", end)
        print("Batch: ",self.sel_arr[start:end])
        print("If Batch Count: ",batch_count," >= slots ",self.batch_slots_to_be_filled, " then 0")
        
            
        # Lower bound for batch percentage criteria
        print("If Batch iterator: ",self.i % self.batch_size," >= rem slots ",self.batch_size - self.batch_slots_to_be_filled, "and Batch Count: ",batch_count," < slots ",self.batch_slots_to_be_filled, " then 1")

        if (self.i % self.batch_size) >= (self.batch_size - self.batch_slots_to_be_filled) and batch_count < self.batch_slots_to_be_filled:
            accept_sample = 1
        
        if(self.i==0):
            accept_sample = random.randint(0, 1)
        
        print("accept_sample: ",accept_sample)
        print("~~~~~~~~~~~~~~~~~~~~~~~~")

        # Returns True if sample is accepted and False otherwise
        return True if accept_sample == 1 else False
    
    def selection_and_train(self,select_samples_percent=0.5):
        # select_samples_percent = perct_sel_smpls 

        self.i = 0 # Dataset Iterator 
        self.mini_batch_slots_to_be_filled = int(select_samples_percent * self.mini_batch_size)
        self.batch_slots_to_be_filled = int(select_samples_percent * self.batch_size)
        
        # Iterate over dataset until it is exhausted
        while(self.i<self.sel_arr.size-1):
        
            # Sample and remove the sample from the dataset (to avoid duplicates in future sampling)
            sample = self.yet_to_train_dataset.sample(n=1)
            self.yet_to_train_dataset.drop(sample.index)
            
            # Perform next steps if sample selection is true
            if self.sample_selection(sample) is True:
                
                if self.sampled_dataset is None:
                    self.sampled_dataset = sample
                else:
                    self.sampled_dataset = self.sampled_dataset.append(sample, ignore_index=True)
               
                self.sel_arr[self.i] = 1
                
                if (self.i % self.batch_size) == 0 and (self.i != 0):
                    lbl, dt = self.prepare_binary(self.sampled_dataset)
                    self.train(dt,lbl)
                
            self.i+=1
        
        lbl, dt = self.prepare_binary(self.sampled_dataset)
        self.train(dt,lbl)   
    
    def _hinge_loss_svm(self,traindata, trainlabel,W,w):
        m =traindata.shape[1]
        # Equation for the regularizer.
        # It is the lambda*(norm2 of W)**2
        # Here "lambda" is a non negative constant
        lambd = cp.Parameter(nonneg=True)

        ## Ideally we will have to try using different values fro "lambda"
        ## For the sake of testing the code, we have set it to 0.01
        ## Do we need to have a lambda?
        lambd = 0.01 
        reg_loss = cp.norm(W,p=2)**2
        
        #hinge loss
        hinge_loss = cp.sum(cp.pos(1-cp.multiply(trainlabel,traindata @ W - w)))
        

        
        #Objective is to minimize reg_loss and hinge_loss
        # objective_func = cp.Minimize(hinge_loss/m + lambd*reg_loss)
        prob = cp.Problem(cp.Minimize(hinge_loss/m + lambd*reg_loss))
        # Now framing the LP, along with the constraints
        return prob

    def _normal_loss_svm(self,traindata,trainlabel, W,w):
        #Constraint
        # For every feature vector traindata[i] and its corresponding label trainlabel[i]:
        # W^T*traindata[i] + w >= 1
        const = [trainlabel[i]*(traindata[i]@ W + w) >= 1 for i in range(traindata.shape[0])]
        ##Check the dimensions in the above constraint equation
        
        #Objective is to minimize reg_loss and hinge_loss
        # objective_func = cp.Minimize(hinge_loss/m + lambd*reg_loss)
        objective_func = cp.Minimize(0.5*cp.norm(W,p=2)**2)
        prob = cp.Problem(objective_func,constraints=const)
        # Now framing the LP, along with the constraints
        return prob

    def train(self,traindata,trainlabel):
        
        #USAGE
        # W, w = train(traindata, trainlabel)

        # m: Number of feature vectors
        # W and w: Weight vector and Bias value respectively
        print(traindata.shape)
        m = traindata.shape[1]
        W = cp.Variable((m,1))
        w = cp.Variable()
        prob = self._hinge_loss_svm(traindata,trainlabel,W,w)
        prob.solve()
        # Solving the problem would give us the optimal values from W and w;
        # which have to be returned, so that we can use them while testing
        ## adding to class variable
        self.w = W
        self.b = w
        

    def f(self,test_input):
        test_val = test_input.dot(self.w.value) -  self.b.value
        if test_val < -1:
            test_val= -1
        elif test_val > 1:
            test_val = 1
        else:
            test_val = 0
        estimated_class = self.classes.get(test_val)
        return estimated_class
    
    def region(self,test_input):
        if self.dist(test_input)<-1:
            return "p1"
        elif self.dist(test_input)>1:
            return "p3"
        else:
            return "p2"
        
    def dist(self,test_input):
        return (test_input.dot(self.w.value) -  self.b.value)
    
    def assess_classifier_performance(self,performance):
        performance = np.asarray(performance)
        correct = (np.count_nonzero(performance)/len(performance))*100
        return correct

    def test(self,dataset_test):
        testlabel,testdata= self.prepare_binary(dataset_test)
        res = []
        performance = 0
        for i in range(testdata.shape[0]):
            result = self.f(testdata[i])
            res.append(result)
            
            actual_class = self.classes.get(int(testlabel[i]))
            ## assessing performance
            if result == actual_class:
                performance += 1
         
        performance /= testlabel.shape[0]
        return res, performance
    
    def train_different_sample_sizes_and_store_performance(self,list_of_samples):
        test_dataset = pd.read_csv('mnist/mnist_test.csv')
        correctness = []
        for percent_sampling in list_of_samples:
            self.selection_and_train(select_samples_percent=percent_sampling)
            performance = self.test(test_dataset)
            correctness.append(self.assess_classifier_performance(performance))
        return correctness

    def plot_classifier_performance_vs_number_of_samples(self):
        pass

    
        


        

