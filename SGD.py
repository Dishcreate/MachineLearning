#correct not accurate
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier


df=pd.read_csv('newtest.csv')
#Y=pd.read_csv('new2.csv')
df1=pd.read_csv('newtest1.csv')
x=df.drop(['tag'],axis=1)
y=df.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X=df1.drop(['tag'],axis=1)
Y=df1.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X_train , X_test , Y_train , Y_test = train_test_split(x,y , random_state=5)

sgdc=SGDClassifier()
sgdc.fit(X_train,Y_train)

print sgdc.score(X_test,Y_test)