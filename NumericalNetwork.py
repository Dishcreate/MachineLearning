#correct
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

df=pd.read_csv('newdata.csv')


x=df.drop(['tag'],axis=1)
y=df.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X_train , X_test , Y_train , Y_test = train_test_split(x,y , random_state=5)
mlpc=MLPClassifier()
mlpc.fit(X_train,Y_train)
y_pre=mlpc.predict(X_test)


print (metrics.accuracy_score(Y_test,y_pre))
