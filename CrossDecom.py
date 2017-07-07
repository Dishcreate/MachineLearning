#correct not accurate
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_decomposition import PLSCanonical
df=pd.read_csv('newdata.csv')
x=df.drop(['tag'],axis=1)
y=df.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X_train , X_test , Y_train , Y_test = train_test_split(x,y , random_state=5)

plsr=PLSRegression()
plsr.fit(X_train,Y_train)

plsc=PLSCanonical()
plsc.fit(X_train,Y_train)

print (plsr.score(X_test,Y_test))
print (plsc.score(X_test,Y_test))
