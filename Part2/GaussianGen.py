import numpy as np
import cvxpy as cp
import pandas as pd
from typing import List
import matplotlib.pyplot as plt


np.random.seed(seed=1)

class GaussGen:

    def __init__(self,mean1:List[int], mean2:List[int], cov1:List[int], cov2:List[int], NoOfSamples:int, FracOfTest:float, class1:int, class2:int):
        self.mean1 = mean1
        self.mean2 = mean2
        self.cov1 = cov1
        self.cov2 = cov2
        self.NoOfSamples = NoOfSamples #Total number of samples to be generated
        self.frac = FracOfTest  #Percentage of entire set should be set aside asd test
        self.class1 = class1
        self.class2 = class2

    def DistGen(self):
        #Output dimension of x1 will be: (NoOfSamplesX2) numpy array
        x1 = np.random.multivariate_normal(self.mean1, self.cov1, self.NoOfSamples)

        #output labels for class 1: we will use -1
        out1 = (np.ones(self.NoOfSamples)*(self.class1)).reshape(self.NoOfSamples,1)  
        dataClass1=np.concatenate((out1,x1),axis=1)
        df1 = pd.DataFrame(dataClass1, columns = ['label','Feature1','Feature2'])
        testdf1 = df1.sample(frac = self.frac)
        #Train set is constructed using the remaing data in df1
        traindf1 = df1.drop(testdf1.index)


        #Output dimension of x2 will be: (NoOfSamplesX2) numpy array
        x2 = np.random.multivariate_normal(self.mean2, self.cov2, self.NoOfSamples)

        #output labels for class 2: we will use 1
        out2 = (np.ones(self.NoOfSamples)*self.class2).reshape(self.NoOfSamples,1)
        dataClass2=np.concatenate((out2,x2),axis=1)
        df2 = pd.DataFrame(dataClass2, columns = ['label','Feature1','Feature2'])
        testdf2 = df2.sample(frac = self.frac)
        #Train set is constructed using the remaing data in df2
        traindf2 = df2.drop(testdf2.index)

        #Entire data set
        fulltrainDf = pd.concat([traindf1,traindf2],axis=0).reset_index(drop=True)
        fulltestDf = pd.concat([testdf1,testdf2],axis=0).reset_index(drop=True)
        #print ("Entire Test set: ", fulltestDf.reset_index(drop=True))
        #print ("Entire Train set: ", fulltrainDf.reset_index(drop=True))
        return fulltrainDf, fulltestDf
       
