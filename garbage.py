import pandas as pd
import cv2
import statistics as st
import scipy.stats as sp
df=pd.read_csv('train.csv')
print(df)

import numpy as np
df=np.matrix(df)
print(np.shape(df))
y1=df[:,1]
#print(y1)


x=np.zeros((1767,7))
y=np.zeros(1767)
x1=df[:,0]
#print(x1)


for i in range(50): 
    y[i]=y1[i]
  #  print (str(x1[i]))
    print(y[i])

    k=np.ravel(x1[i])
    a=np.ravel(k)

    fn='Garbage classification/'+str(a[0])
    print(fn)
    I=cv2.imread(fn)
  #  print(I)
  #  cv2.imshow("",I)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    r=I
    r=np.ravel(r)
   # print(r)
    f1=np.median(r)
   # print(f1)
    f2=np.mean(r)
   # print(f2)
    f3=st.mode(r)
   # print(f3)
    f4=np.std(r)
   # print(f4)
    f5=np.var(r)
   # print(f5)
    f6=sp.skew(r)
   # print(f6)
    f7=sp.kurtosis(r)
   # print(f7)
    f=[f1,f2,f3,f4,f5,f6,f7]
   # print(f)
   # print(np.shape(f))

    x[i,:]=f
   # y[i]=i
print(np.shape(x))
print(np.shape(y))

#from sklearn.tree import DecisionTreeClassifier
#classifier=DecisionTreeClassifier()
#classifier.fit(x,y)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(x,y)
accuracy=classifier.score(x,y)
print('accuracy=',accuracy)

I=cv2.imread('Garbage classification/glass380.jpg')
#print(I)
cv2.imshow("",I)
cv2.waitKey(1000)
cv2.destroyAllWindows()

r=I
r=np.ravel(r)
#print(r)
f1=np.median(r)
#print(f1)
f2=np.mean(r)
#print(f2)
f3=st.mode(r)
#print(f3)
f4=np.std(r)
#print(f4)
f5=np.var(r)
#print(f5)
f6=sp.skew(r)
#print(f6)
f7=sp.kurtosis(r)
#print(f7)
f=[f1,f2,f3,f4,f5,f6,f7]
#print(f)

f=[f1,f2,f3,f4,f5,f6,f7]    
f=np.matrix(f)
res=classifier.predict(f)
#print(res-1)
result=int(np.floor(abs(res)))
print(result)
#print(y1[result])


if result==1:
    print('glass')
elif result==2:
    print('paper')    
elif result==3:
    print('cardboard')
elif result==4:
    print('plastic')
elif result==5:
    print('metal')
elif result==6:
    print('trash')

