#error

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


df=pd.read_csv('newtest.csv')
df1=pd.read_csv('newtest1.csv')
x=df.drop(['tag'],axis=1)
y=df.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X=df1.drop(['tag'],axis=1)
Y=df1.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X_train , X_test , Y_train , Y_test = train_test_split(x,y , random_state=5)

ir=IsotonicRegression()
ir.fit(X_train,Y_train)


print ir.score(X_test,Y_test)
