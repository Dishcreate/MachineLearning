#correct
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

df=pd.read_csv('newdata.csv')
x=df.drop(['tag'],axis=1)
y=df.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X_train , X_test , Y_train , Y_test = train_test_split(x,y , random_state=5)

rfc=RandomForestClassifier()
rfc.fit(X_train,Y_train)

etc=ExtraTreesClassifier()
etc.fit(X_train,Y_train)

print (metrics.accuracy_score(Y_test,rfc.predict(X_test)))
print (metrics.accuracy_score(Y_test,etc.predict(X_test)))