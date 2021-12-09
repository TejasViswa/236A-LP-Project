import numpy as np
import pandas as pd
import cvxpy as cp
import random
import numpy.linalg as la

from pandas.io.pytables import performance_doc
pd.options.mode.chained_assignment = None  # default='warn'   # To disable the warning messages that we were getting


class MyClassifier_25:  

    def __init__(self,dataset,class1:int,class2:int,lambd:float) -> None:
        self.w = None
        self.b = None
        self.classes = { 1 : class1, -1: class2, 0:None}
        self.dataset_train = dataset
        self.lambd = lambd

	#Centroids of each cluster
        self.c1 = None
        self.c2 = None
        self.c3 = None

        #data prep
        #self.trainlabel,self.traindata = self.prepare_binary(self.dataset_train)
        
    
    def prepare_binary(self,dataset):
        #USAGE    
        # Since we have to deal with a binary classifier to diffrentiate between digits 7 and 1, 
        # we choose only those examples.
        # If asked to train a classifier on any other pair a, b (say),
        # please pass the right arguments to the following function as follows:
        # trainlabel, traindata, = prepare_binary(a,b)


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
        targetDf=pd.DataFrame(trainlabel,columns=['label'])
        
        dataTargetDf = pd.concat([trainDf, targetDf[['label']]], axis = 1)
        ##If randomizing the order, should we use the dataframe 'finalDf'?
        return dataTargetDf


    def subset(self,dataTargetDf, subsetfrac:float):
        
        # Usage: If 20% of the data is to be randomly selected
        # subsetDf = subset(dataTargetDf, 0.2)
        
        return dataTargetDf.sample(frac = subsetfrac)


    #PROJECT PART 2: Reducing the size of the train set
    def ILP(self, dataset):
        
        df = dataset  #path to the entire data csv file

        class1 = self.classes[1]
        class2 = self.classes[-1]

        label1data = df.loc[df['label'] == class1] #dataframe with all rows corresponding to label1
        label2data = df.loc[df['label'] == class2] #dataframe all rows corresponding to label7
        label3data = df.loc[df['label'].isin([class1,class2])] #Dataframe with both label1 and label
    
    
        #Number of samples in each class
        train1_count = len(label1data) 
        train2_count = len(label2data)
        train3_count = len(label3data)
        
        #Extracting only the feature vectors by dropping the corrsponding labels
        self.train1_data = label1data.drop(columns = ['label']) #dropping the column1 consisting of labels
        self.train2_data = label2data.drop(columns = ['label'])
        self.train3_data = label3data.drop(columns = ['label'])

        #Calculating the centroid for each of the groups
        #c1 and c2: Centroid for label1 and label2 respectively
        #c3: cebtroid of the entire dataset (label1 and label2 combined)
        self.c1 = self.train1_data.sum()/train1_count #centroid = sum of each component of all vectors/number
        self.c2 = self.train2_data.sum()/train2_count
        self.c3 = self.train3_data.sum()/train3_count
       
        #The distance between each of the centroids
        #dist12 = la.norm(c1-c2) #dist between centroids of 1 and 2
        #dist13 = la.norm(c1-c3)
        #dist23 = la.norm(c2-c3)

        #select_func() expects only the features and not labels;
        #hence we drop the labels
             
        selected_train1_logic1 = self.select_func(label1data.drop(columns=['label']), 1)
        selected_train2_logic1 = self.select_func(label2data.drop(columns=['label']), 2)
        

        #We then append the corresponding labels to the selected train set
        #This is ensure that we do not mess up with label order, i.e. feature vectors and
        # their corresponding labels match
        selected_train1_logic1_Df = pd.concat([label1data.loc[selected_train1_logic1.index,'label'], selected_train1_logic1], axis = 1)
        selected_train2_logic1_Df = pd.concat([label2data.loc[selected_train2_logic1.index,'label'], selected_train2_logic1], axis = 1)
        
        #The reduced dataframe with both the classes:
        selected_Df_logic1 = pd.concat([selected_train1_logic1_Df, selected_train2_logic1_Df], axis = 0)
        
        return selected_Df_logic1

    def func_min_dist_point_to_label2(self, point):
        dist_min = np.min(la.norm(np.subtract(self.train2_data.to_numpy(),np.array(point)),axis=1))
        return dist_min

    def func_min_dist_point_to_label1(self, point):     
        dist_min = np.min(la.norm(np.subtract(self.train1_data.to_numpy(),np.array(point)),axis=1))
        return dist_min

    def dist_to_centroid1(self, point):
        z = np.subtract(np.array(point[:len(self.c1)]),np.array(self.c1))
        return(la.norm(z))

    def dist_to_centroid2(self,point):
        z = np.subtract(np.array(point[:len(self.c2)]),np.array(self.c2))
        return(la.norm(z))
    
    def dist_to_centroid3(self, point):
        z = np.subtract(np.array(point[:len(self.c3)]),np.array(self.c3))
        return(la.norm(z))

    def select_func(self, datadf, label):
        if label == 1:
            # function calculating the minimum distance between a cluster 1 point and cluster 1 centroid
            g = self.dist_to_centroid1
            # function calculating the minimum distance between a cluster 1 point and cluster 2 centroid
            h = self.dist_to_centroid2
        else:
            # function calculating the minimum distance between a cluster 2 point and cluster 2 centroid
            g = self.dist_to_centroid2
            # function calculating the minimum distance between a cluster 2 point and cluster 1 centroid
            h = self.dist_to_centroid1
            
        # Function to find the distance between all the points and the centroid (c3) of the entire dataset.
        # Since this function is common to both the classes, we have written it outside the if loop.
        y = self.dist_to_centroid3

        # LOGIC 1
        # Please note that since we were trying multiple logics, we have "_logic1" given to different variables. 
        # This code has only the best working logic (logic 1) retained. 
        # We have retained "_logic1" to avoid any naming related mismatch in other sections of the code.
     
        x = datadf.copy() 
        x['Dist_to_c3'] = datadf.apply(y,axis=1)
        x['Dist_other_centroid'] = datadf.apply(h,axis=1)
        x['Dist_own_centroid'] = datadf.apply(g,axis=1)
        
        # Select only those samples that are closer to c3 than the centroid of the other class and to it's own centroid. 
        selected_train = x.loc[(x['Dist_to_c3'] < x['Dist_other_centroid']) & (x['Dist_to_c3'] < x['Dist_own_centroid'])]
        selected_train_logic1 = selected_train.drop(columns = ['Dist_own_centroid', 'Dist_other_centroid','Dist_to_c3'])
        
        return selected_train_logic1

    # Function to implement K nearest neighbors.
    def dis_sim_ngbr(self,dataset,no_of_ngbr=5,no_of_dissim=1): # Discard points with similar neighbours
        dissim_dataset = dataset
        i = dataset.index[0]
        dt_done = []
        while len(dt_done) < len(dataset):
        
            dist_arr = self.sorted_dist_from_other_pts(dissim_dataset,i)
            dist_arr = dist_arr.iloc[0:no_of_ngbr]

            dis = 0
            for it in dist_arr.index:
                if dissim_dataset.loc[[it]]['label'].values[0] != dissim_dataset.loc[[i]]['label'].values[0]:
                    dis+=1

            if dis < no_of_dissim:
                dissim_dataset = dissim_dataset.drop([i])
    
            dt_done.append(i)
            for i in dissim_dataset.index:
                if i in dt_done:
                    continue
                else:
                    break
            
        return dissim_dataset


    # Function related to the above K-NN function above.
    def sorted_dist_from_other_pts(self,dataset,i):
        X = dataset.drop(columns=['label'])
        ddd = X.sub(X.loc[i])
        ddd = np.sqrt(np.square(ddd).sum(axis=1))
        ddd = ddd.sort_values(ascending=True)
        return ddd

    def _hinge_loss_svm(self,traindata, trainlabel,W,w):
        m =traindata.shape[1]
        # Equation for the regularizer.
        # It is the lambda*(norm2 of W)**2
        # Here "lambd" is a non negative constant
        lambd = cp.Parameter(nonneg=True)

        lambd = self.lambd
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
        #print(traindata.shape)
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
        if test_val < 0:
            test_val= -1
        #elif test_val > 1:
        #    test_val = 1
        else:
            test_val = 1
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
        #print ("Type of the performance vector:", performance)
        performance = np.asarray(performance)
        correct = (np.count_nonzero(performance)/len(performance))*100
        return correct

    def test(self,dataset_test):
        testlabel,testdata= self.prepare_binary(dataset_test)
        res = []
        performance = []
        print("Started to test!!!")
        for i in range(testdata.shape[0]):
            result = self.f(testdata[i])
            res.append(result)

            actual_class = self.classes.get(int(testlabel[i]))

            if result == actual_class: #[0]:
                performance.append(1)
            else:
                performance.append(0)
        
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


