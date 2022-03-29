#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[41]:


df= pd.read_csv("creditcard.csv")
df.head()


# In[42]:


df.shape


# In[43]:


df.isnull().sum()


# In[44]:


sns.heatmap(df.isnull(),yticklabels=False,cmap="rainbow")


# In[45]:


df['Class'].value_counts()


# In[46]:


X=df.drop(columns='Class')


# In[47]:


Y=df['Class']
Y


# In[48]:


from imblearn.over_sampling import SMOTE


# In[49]:


UtoB=SMOTE(sampling_strategy='auto',random_state=35,k_neighbors=4)
X1,Y1=UtoB.fit_resample(X,Y)


# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


X_train,X_test,Y_train,Y_test=train_test_split(X1,Y1,test_size=0.33,random_state=35)


# In[52]:


from sklearn.linear_model import LogisticRegression


# In[53]:


lr=LogisticRegression(solver='liblinear')


# In[54]:


lr.fit(X_train,Y_train)


# In[55]:


from sklearn.metrics import accuracy_score


# In[56]:


y_pred=lr.predict(X_test)
score=accuracy_score(y_pred,Y_test)
score


# In[57]:


from sklearn.metrics import confusion_matrix


# In[58]:


cm=confusion_matrix(Y_test, y_pred, labels=lr.classes_)
cm


# In[59]:


from sklearn.metrics import ConfusionMatrixDisplay


# In[60]:


cmd=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=lr.classes_)
cmd.plot()


# In[61]:


from sklearn.metrics import recall_score


# In[62]:


print(recall_score(y_pred,Y_test,average='binary'))


# In[63]:


from sklearn.metrics import precision_score


# In[64]:


print(precision_score(Y_test,y_pred,average='binary'))


# In[65]:


from sklearn.metrics import f1_score


# In[66]:


print(f1_score(y_pred,Y_test,average='binary'))


# In[67]:


from sklearn.metrics import classification_report


# In[68]:


print(classification_report(Y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




