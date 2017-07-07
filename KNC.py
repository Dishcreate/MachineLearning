from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import pandas as pd


df=pd.read_csv('newdata.csv')
x=df.drop(['tag'],axis=1)
y=df.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X_train , X_test , Y_train , Y_test = train_test_split(x,y , random_state=4)
knc=KNeighborsClassifier(n_neighbors=5)
knc.fit(x,y)
y_pre=knc.predict(X_test)
print (y_pre)
print (metrics.accuracy_score(Y_test,y_pre))