#correct
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

df=pd.read_csv('newdata.csv')
x=df.drop(['tag'],axis=1)
y=df.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X_train , X_test , Y_train , Y_test = train_test_split(x,y , random_state=5)

dtc=DecisionTreeClassifier()
dtc.fit(X_train,Y_train)
y_pre1=dtc.predict(X_test)

dtr=DecisionTreeRegressor()
dtr.fit(X_train,Y_train)


print (metrics.accuracy_score(Y_test,y_pre1))
