#crrect
import pandas as pd
from sklearn.svm import SVC
import time



start = time.time()
df1= pd.read_csv('newdata.csv')

df1 = df1.dropna()  # remove rows with missing values
X=df1.drop(["tag"], axis=1)
Y=df1.drop(["kx","ky","kz","wa","wb","wc","wd","we","wf"],axis=1)



clf=SVC()


from sklearn.cross_validation import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split(X,Y, random_state=3)

clf.fit(X, Y)

#A=clf1.kneighbors_graph(X_train)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
Y_pre=clf.predict(X_test)

#print Y_true1
#accu=clf1.score(Y_true,Y_pre)
print (accuracy_score(Y_test,Y_pre))


#print predict




