
def prepare_dataset(dataset_path):
    '''
    Read a comma separated text file where
	- the first field is a ID number
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued
    Return two numpy arrays X and y where
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'
    @param dataset_path: full path of the dataset text file
    @return
	X,y
    '''
    ##         "INSERT YOUR CODE HERE"
    import numpy as np
    import pandas as pd
    path = dataset_path
    df = pd.read_csv(path)  # read a comma separated text file
    df = df.fillna(df)  # filter out rows which has blank values
    dataset1 = df.values[:, 9:10]
    dataset2 = df.values[:, 0:9]
    y = np.array(dataset1)  # set numpy array for y
    X = np.array(dataset2)  # set numpy array for X
    return X, y
    #print(X)
    # raise NotImplementedError()


def build_tain_data(X_data, y_data):
    from sklearn.model_selection import train_test_split
    X = X_data
    y = y_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
    return X_train, X_test, y_train, y_test


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training):
    '''
    Build a Naive Bayes classifier based on the training set X_training, y_training.
    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    from sklearn.naive_bayes import GaussianNB
    X = X_training
    y = y_training
    clf = GaussianNB()
    clf.fit(X, y)
    return clf
    # raise NotImplementedError()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training):
    '''
    Build a Decision Tree classifier based on the training set X_training, y_training.
    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    from sklearn import tree
    X = X_training
    y = y_training
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    return clf
    # raise NotImplementedError()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.
    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    from sklearn.neighbors import NearestNeighbors
    X = X_training
    y = y_training
    clf = NearestNeighbors()
    clf.fit(X, y)
    return clf
    # raise NotImplementedError()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def build_KNC_classifier(X_training,y_training):
    '''
        Build a K-Nearest Classifier Machine classifier based on the training set X_training, y_training.
        @param
    	X_training: X_training[i,:] is the ith example
    	y_training: y_training[i] is the class label of X_training[i,:]
        @return
    	clf : the classifier built in this function
        '''
    ##         "INSERT YOUR CODE HERE"
    from sklearn.neighbors import KNeighborsClassifier
    X = X_training
    y = y_training
    clf = KNeighborsClassifier()
    clf.fit(X, y)
    return clf


def build_KNR_classifier(X_training,y_training):
    '''
        Build a K-Nearest Regressor Machine classifier based on the training set X_training, y_training.
        @param
    	X_training: X_training[i,:] is the ith example
    	y_training: y_training[i] is the class label of X_training[i,:]
        @return
    	clf : the classifier built in this function
        '''
    ##         "INSERT YOUR CODE HERE"
    from sklearn.neighbors import KNeighborsRegressor
    X = X_training
    y = y_training
    clf = KNeighborsRegressor()
    clf.fit(X, y)
    return clf




def build_SVM_classifier(X_training, y_training):
    '''
    Build a Support Vector Machine classifier based on the training set X_training, y_training.
    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    from sklearn import svm
    X = X_training
    y = y_training
    clf = svm.SVC()
    clf.fit(X, y)
    return clf
    # raise NotImplementedError()


def accuracy_test(clf, X_testing,Y_testing):
    from sklearn import metrics




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    # call your functions here

    X_data_set, y_data_set = prepare_dataset("newdata.csv")
    X_training, X_testing, y_training, y_testing = build_tain_data(X_data_set, y_data_set)
    NBClf=build_NB_classifier(X_training, y_training)
    DTClf=build_DT_classifier(X_training, y_training)
    NNClf=build_NN_classifier(X_training, y_training)
    SVMClf=build_SVM_classifier(X_training, y_training)
    KNCClf=build_KNC_classifier(X_training,y_training)
    KNRClf=build_KNR_classifier(X_training,y_training)


