#!/usr/bin/env python
# coding: utf-8

# ## <font color = red >2.1  クラス分類と回帰</font>

# ## <font color = red>2.2  汎化、過剰適合（過学習）、適合不足</fond>

# ### <font color = blue>2.2.1  モデルの複雑さとデータセットの大きさ</font>

# ## <font color = red>2.3  教師あり機械学習アルゴリズム</font>

# ### <font color = blue>2.3.1  サンプルデータセット<font>

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import mglearn
import pandas as pd


# In[2]:


# generate dataset
from sklearn.datasets import make_blobs
X1, y1 = mglearn.datasets.make_forge()
X, y = make_blobs()
# plot dataset
plt.figure(1)
mglearn.discrete_scatter(X[:,  0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc = 4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
print("X.shape: {}".format(X.shape))

plt.figure(2)
mglearn.discrete_scatter(X1[:,  0], X1[:, 1], y1)
plt.legend(["Class 0", "Class 1"], loc = 4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
print("X1.shape: {}".format(X1.shape))


# In[3]:


X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()


# In[4]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))


# In[5]:


print("Shape of cancer data:{} \n".format(cancer.data.shape))


# In[6]:


print("Sample counts per class:\n{}".format(
    {
        n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))
    }
))


# In[7]:


print("Feature names:\n{}".format(cancer.feature_names))


# In[8]:


from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))


# In[9]:


X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))


# ## <font color = blue>2.3.2  k-最近傍法（K-NN）</font>

# ### 2.3.2.1  k-最近傍法によるクラス分類

# In[10]:


# knnの中 k=1の状況
mglearn.plots.plot_knn_classification(n_neighbors=1)


# In[11]:


# knnの中 k=3の状況
mglearn.plots.plot_knn_classification(n_neighbors = 3)


# In[12]:


from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


# In[13]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)


# In[14]:


clf.fit(X_train, y_train)


# In[15]:


print("Test set prediction: {}".format(clf.predict(X_test)))


# In[16]:


print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))


# ### 2.3.2.2  KneighborsClassifierの解析

# In[17]:


fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    # fitメソッドは自分自身を返すので､一行で
    # インスタンスを生成してfitすることができる
    clf = KNeighborsClassifier(n_neighbors = n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")

axes[0].legend(loc=3)


# In[18]:


from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66
)

training_accuracy = []
test_accuracy = []

# try n_neighbor from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record testing set accuracy
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="testing accuracy")
plt.xlabel("Accuracy")
plt.ylabel("n_neighbors")
plt.legend()


# ### 2.3.2.3  k-近傍回帰

# In[19]:


mglearn.plots.plot_knn_regression(n_neighbors=1)


# In[20]:


mglearn.plots.plot_knn_regression(n_neighbors=3)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)

# waveデータセットをテストセットに分割
X_train, X_test, y_train, y_test


# In[22]:


try:   
    get_ipython().system('jupyter nbconvert --to python 2章 教師あり学習.ipynb')
    # python即转化为.py，script即转化为.html
    # file_name.ipynb即当前module的文件名
except:
    pass


# In[ ]:




