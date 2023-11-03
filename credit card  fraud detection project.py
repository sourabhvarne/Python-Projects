import pandas as pd
import numpy as np

df=pd.read_csv('creditcard.csv')
print(df)

df1=pd.set_option('display.max_columns',31)
#print(df1)

df2=df.head()
#print(df2)

df3=df.tail()
#print(df3)

#df4=df.info()
#print(df4)

##checking null value

df5=df.isnull()
#print(df5)

##transaction detils

df6=df['Class'].value_counts()
print(df6)

#o--normal transaction
#1--fraud transaction

import matplotlib.pyplot as plt

labels = ["normal", "Fraud"]
count_classes = df.value_counts(df['Class'], sort= True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Visualization of Labels")
plt.ylabel("Count")
plt.xticks(range(2), labels)
#plt.show()
'''
#separate data for analysis
normal=df[df.Class==0]
fraud=df[df.Class==1]
print(np.shape(normal))
print(np.shape(fraud))

df7=normal.Amount.describe()
#print(df7)

df8=fraud.Amount.describe()
#print(df8)

#compare data by mean
df9=df.groupby('Class').mean()
#print(df9)

#normal_sample=fraud_sample(n=492)

df10=pd.concat([normal,fraud],axis=0)
#print(df10)
'''
df=np.matrix(df)
print(np.shape(df))
y=df[:,30]
#print(y)

x=df[:,:30]
#print(x)
print(np.shape(x))

y=np.ravel(y)
print(np.shape(x))
print(np.shape(y))

from sklearn import linear_model
mdl=linear_model.LogisticRegression()
mdl.fit(x,y)
accuracy=mdl.score(x,y)
print('accuracy=',accuracy)


q=[0,-1.359807134   ,-0.072781173,2.536346738,	1.378155224 ,-0.33832077,0.462387778,0.239598554,0.098697901,0.36378697	,0.090794172,	-0.551599533,	-0.617800856,-0.991389847,-0.311169354,1.468176972  ,-0.470400525,0.207971242,0.02579058,	0.40399296,	0.251412098,	-0.018306778,	0.277837576,	-0.11047391,	0.066928075,0.128539358 ,-0.189114844,0.133558377 ,-0.021053053,149.62
 ]

q=np.matrix(q)
res=mdl.predict(q)
result=int(np.floor(abs(res)))
print(result)
if result==0:
           print('normal transaction')
elif result==1:
           print('fraud transaction')


