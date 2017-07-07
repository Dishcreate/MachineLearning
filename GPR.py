#correct
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier

df=pd.read_csv('newdata.csv')
x=df.drop(['tag'],axis=1)
y=df.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X=df.drop(['tag'],axis=1)
Y=df.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X_train , X_test , Y_train , Y_test = train_test_split(x,y , random_state=5)

knn=GaussianProcessClassifier()
knn.fit(X_train,Y_train)
y_pre=knn.predict(X_test)


print (metrics.accuracy_score(Y_test,y_pre))