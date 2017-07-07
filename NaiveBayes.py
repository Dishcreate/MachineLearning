#correct
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


df=pd.read_csv('newdata.csv')
x=df.drop(['tag'],axis=1)
y=df.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X_train , X_test , Y_train , Y_test = train_test_split(x,y , random_state=5)

gnb=GaussianNB()
gnb.fit(X_train,Y_train)
y_pre=gnb.predict(X_test)
print(y_pre)
print(Y_test)

print (metrics.accuracy_score(Y_test,y_pre))