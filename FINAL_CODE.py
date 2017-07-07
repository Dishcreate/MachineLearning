#crrect
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import serial

port = serial.Serial('COM12',115200)

#print out
def Output():
    while True:
        #for list in range(6):
        out = port.readline()
        return out

a =Output()
#print a.split()
#for i in range(6):
b=map(int, a.split(','))
print (b)
#print b
c=np.array([b])
#print a

df=pd.read_csv('newtest.csv')
#df2=pd.read_csv('testpoint2.csv',header=None)

#Y=pd.read_csv('new2.csv')
df1=pd.read_csv('newtest1.csv')
x=df.drop(['tag'],axis=1)
y=df.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X=df1.drop(['tag'],axis=1)
Y=df1.drop(['kx','ky','kz','wa','wb','wc','wd','we','wf'],axis=1)
X_train , X_test , Y_train , Y_test = train_test_split(x,y , random_state=5)

def KNR():
    from sklearn.neighbors import KNeighborsRegressor
    knn=KNeighborsRegressor(n_neighbors=5)
    knn.fit(x.values,y.values)
    y_pre=knn.predict(b)
    return y_pre
#print KNR()

def KNC():
    from sklearn.neighbors import KNeighborsClassifier
    knc=KNeighborsClassifier()
    knc.fit(x.values,y.values)
    y_pre=knc.predict(b)
    return y_pre

def SVCMOde():
    from sklearn.svm import SVC
    svc=SVC()
    svc.fit(x.values,y.values)
    y_pre=svc.predict(b)
    return y_pre


# def kMean():
#     from sklearn.cluster import KMeans
#     km=KMeans(n_clusters=10)
#     km.fit(x.values)
#     y_pre=km.predict(b)
#     return y_pre
#
# print kMean()

def LR():
    from sklearn.linear_model import LinearRegression
    lm=LinearRegression()
    lm.fit(x.values,y.values)
    y_pre=lm.predict(b)
    return y_pre

#print LR()

def DTC():
    from sklearn.tree import DecisionTreeClassifier
    dtc=DecisionTreeClassifier()
    dtc.fit(x.values,y.values)
    y_pre=dtc.predict(b)
    return y_pre

def DTR():
    from sklearn.tree import DecisionTreeRegressor
    dtr=DecisionTreeRegressor()
    dtr.fit(x.values,y.values)
    y_pre=dtr.predict(b)
    return y_pre

#print KNR(),KNR(),LR(),SVCMOde(),DTC(),DTR()



