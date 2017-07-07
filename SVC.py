#correct
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.svm import LinearSVC ,SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('newtest.csv')
#df2=pd.read_csv('newtest.csv')
df1=pd.read_csv('newtest1.csv')
#df2=pd.read_csv('testpoint2.csv',header=None)
x=df.drop(['tag'],axis=1)
x = df.fillna(x)
x1=df.drop(['kx','ky','kz','wa','wb','wc','we','tag'],axis=1)

X_plot=x1.values[:, :2]

y=df.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
y=df.fillna(y)
X=df1.drop(['tag'],axis=1)
Y=df1.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X_train , X_test , Y_train , Y_test = train_test_split(x,y , random_state=5)

svm=SVC()
svm.fit(x,y)
#y_pre=svm.predict([4, -10, 0, 50, 0, 0, 50, 0, 50])
print (y_pre)

print metrics.accuracy_score(Y_test,y_pre)
