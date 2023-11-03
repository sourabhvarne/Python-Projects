import pandas as pd
import numpy as np

df=pd.read_csv('heart.csv')
print(df)

'''
df=df.isnull()
print(df)

'''
#statical measure about the data
'''
df=df.describe()
print(df)
'''
#distribution of target varible
'''
df=df['target'].value_counts()
print(df)
'''
#1=defective heart
#0=Healty heart

df=np.matrix(df)
print(np.shape(df))
y=df[:,13]
#print(y)

x=df[:303,:13]
print(x)
print(np.shape(x))

import matplotlib.pyplot as plt

'''
a=df[:,0]
plt.plot(a,marker='*',color='red')
plt.show()
'''

b=df[:,1]
plt.plot(b,marker='*',color='black')
plt.show()




'''
y=np.ravel(y)
print(np.shape(x))
print(np.shape(y))

from sklearn import linear_model
mdl=linear_model.LogisticRegression()
mdl.fit(x,y)
accuracy=mdl.score(x,y)
print('accuracy=',accuracy)



q=[62,0,2,130,263,0,1,97,0,1.2,1,1,3]


q=np.matrix(q)
res=mdl.predict(q)
result=int(np.floor(abs(res)))
print(result)
if result==0:
           print('Healthy Heart')
elif result==1:
           print('Defective Heart')



'''












