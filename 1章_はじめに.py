#!/usr/bin/env python
# coding: utf-8

# import
# --

# In[1]:


import numpy as np
import pandas as pd
import mglearn as mg


# 1.7.1  データを読む
# --

# In[ ]:


from sklearn.datasets import load_iris
iris_dataset = load_iris()


# In[2]:


print(type(iris_dataset))


# which is similar to a dictionary. It also contains keys and values

# Keys
# --

# In[3]:


print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))


# In[4]:


print(iris_dataset['DESCR'][:193] +"\n...")


# Target names
# --

# In[76]:


print("Target names: {}".format(iris_dataset.target_names))


# Feature names
# --

# In[6]:


print("Feature names: {}".format(iris_dataset['feature_names']))


# type of data
# --

# In[7]:


print("type of data: {}".format(type(iris_dataset['data'])))


# shape of data
# --

# In[8]:


print("shape of data: {}".format(iris_dataset['data'].shape))


# First five columns of data
# --

# In[91]:


print("First five columns of data:\n{}".format(iris_dataset.data[:5]))


# Type of target
# --

# In[10]:


print("Type of target: {}".format(type(iris_dataset['target'])))


# Target values
# --

# In[11]:


print("Target:\n{}".format(iris_dataset['target']))


# 0 means setosa, 1 means versicolor, and 2 means virginica

# 1.7.2  成功度合いの測定：訓練データとテストデータ
# --

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)


# In[13]:


print("X_train shape:{}".format(X_train.shape))
print("X_test shape:{}".format(X_test.shape))


# In[14]:


print("y_train shape:{}".format(y_train.shape))
print("y_test shape:{}".format(y_test.shape))


# 1.7.3  データを観察する
# --

# In[15]:


iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset["feature_names"])
grr = pd.plotting.scatter_matrix(
    iris_dataframe,
    c=y_train,
    figsize=(15, 15),
    marker='o',
    hist_kwds={'bins': 20},
    s=60,
    alpha=0.8,
    cmap=mg.cm3
    )


# 1.7.4  モデル：k−最近傍法
# --

# In[16]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)


# 1.7.5  予測を行う
# --

# In[17]:


X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:{}".format(X_new.shape))

prediction = knn.predict(X_new)
print("prediction:{}".format(prediction))
print("predicted target name:{}".format(
    iris_dataset['target_names'][prediction]
))


# 1.7.6  モデルの評価
# --

# In[18]:


y_pred = knn.predict(X_test)


# In[19]:


print("Test set predictions:\n {}".format(y_pred))


# In[20]:


print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))


# In[24]:


print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# 1.8  訓練と評価を行うために必要な最小の手順：
# --

# In[40]:


X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)


# In[41]:


knn = KNeighborsClassifier(n_neighbors = 1)


# In[42]:


knn.fit(X_train, y_train)


# In[43]:


print("Test set score: {:.3f}".format(knn.score(X_test, y_test)))


# k-fold 交差検証
# --

# In[83]:


from sklearn.model_selection import KFold
kfold = KFold(n_splits = 6, shuffle=True)


# In[84]:


from sklearn.model_selection import cross_val_score


# In[85]:


print("Cross-validation scores:\n {}".format(cross_val_score(knn, iris_dataset.data, iris_dataset.target, cv=kfold)))


# In[48]:


accuracy = []
for i in range(11):
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = i)
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X_train, y_train)
    KNeighborsClassifier(n_neighbors=1)
    score = knn.score(X_test, y_test)
    print("Test set {} score: {:.5f}".format(i, score))
    accuracy.append(score)

print("{} distributions' average score is :{}".format(11, np.mean(accuracy))) 

