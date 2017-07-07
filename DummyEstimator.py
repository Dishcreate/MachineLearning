#correct
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.dummy import DummyClassifier

df=pd.read_csv('newtest.csv')
#Y=pd.read_csv('new2.csv')
df1=pd.read_csv('newtest1.csv')
x=df.drop(['tag'],axis=1)
y=df.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X=df1.drop(['tag'],axis=1)
Y=df1.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X_train , X_test , Y_train , Y_test = train_test_split(x,y , random_state=5)

dc=DummyClassifier(strategy='most_frequent',random_state=0)
dc.fit(X_train,Y_train)

DummyClassifier(constant=None, random_state=0, strategy='most_frequent')
dc.score(X_test, Y_test)

