#error
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import numpy as np
import pandas as pd


df=pd.read_csv('newtest.csv')
#Y=pd.read_csv('new2.csv')
df1=pd.read_csv('newtest1.csv')
x=df.drop(['tag'],axis=1)
y=df.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X=df1.drop(['tag'],axis=1)
Y=df1.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X_train , X_test , Y_train , Y_test = train_test_split(x,y , random_state=4)

nn=NearestNeighbors()
nn.fit(x,y)

#print metrics.accuracy_score(Y_test,y_pre)