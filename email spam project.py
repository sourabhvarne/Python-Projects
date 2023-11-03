import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

df=pd.read_csv('mail_data.csv')
print(df)

df1=df.head()
print(df1)

df2=df.isnull()
#print(df2)

df3=df['Category'].value_counts()
print(df3)


labels = ["ham", "spam"]
count_category = df.value_counts(df['Category'], sort= True)
count_category.plot(kind = "bar", rot = 0)
plt.title("Visualization of graph")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()

##separate data text and label
x=df['Message']
y=df['Category']
#print(x)
#print(y)
print(np.shape(x))
print(np.shape(y))


for i in range(5572):
 
    if y[i]=='spam':
        y[i]=0
    elif y[i]=='ham':
        y[i]=1
print(y)
print(np.shape(y))






x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
print(np.shape(x))
print(np.shape(x_train))
print(np.shape(x_test))

##feature Extraction to convert text data to numeric value
feature_extraction=CountVectorizer(min_df=1,stop_words='english',lowercase='True')
x_train_features=feature_extraction.fit_transform(x_train)
x_test_features=feature_extraction.fit_transform(x_test)
#print(x_train_features)

y_train=y_train.astype('int')
y_test=y_test.astype('int')
#print(y_train)
#print(y_test)

print('######################')

from sklearn import linear_model
mdl=linear_model.LogisticRegression()
mdl.fit(x_train_features,y_train)
accuracy=mdl.score(x_train_features,y_train)
print('accuracy for train data=',accuracy)

mdl.fit(x_test_features,y_test)
accuracy=mdl.score(x_test_features,y_test)
print('accuracy for test data=',accuracy)


q=["WINNER!! As a valued network customer you have been selected to receivea Â£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]
q=feature_extraction.transform(q)
res=mdl.predict(q)
result=int(np.floor(abs(res)))



print('#########################')
print(result)
if result==0:
           print('spam Mail')
elif result==1:
           print('ham Mail')





























































