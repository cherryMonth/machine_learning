#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
# 网址可以直接复制
data = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# 取特征值
x = data[['pclass','age','sex']]
# 取目标值
y = data[['survived']]
x['age'].fillna(x['age'].mean(),axis=0,inplace=True)

"""
train_test_split 为划分数据集，返回四个结果
第一个为训练特征矩阵，第二个为训练标签矩阵
第三个为测试特征矩阵，第四个为测试标签矩阵

test_size为两个数据集的比例
random_state 为随机数种子，不同的种子得到的结果是不同的
如果种子相同且不为零，那么结果是相同的，如果为零那么每次结果都是随机的
"""

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


"""
参数orient可以是字符串{'dict', 'list', 'series', 'split', 'records', 'index'}中的
任意一种来决定字典中值的类型
字典dict（默认）：类似于{列：{索引：值}}这样格式的字典
列表list：类似于{列：[值]}这种形式的字典
序列series：类似于{列：序列（值）}这种形式的字典
分解split：类似于{索引：[索引]，列：[列]，数据：[值]}这种形式的字典
记录records：类似于[{列：值}，...，{列：值}]这种形式的列表
索引index：类似于{索引：{列：值}}这种形式的字典
"""

x_train = x_train.to_dict(orient='records')     # 加了orient="records"  以行操作
x_test = x_test.to_dict(orient='records')

"""
DictVectorizer作用是把字典类型的列表向量化

[{'sex': 'female', 'age': 28.0, 'pclass': '2nd'}, 
{'sex': 'female', 'age': 28.0, 'pclass': '2nd'}]

这种格式变成

[[47.          1.          0.          0.          1.          0.        ]
 [64.          1.          0.          0.          1.          0.        ]
 [31.19418104  0.          0.          1.          1.          0.        ]
 [31.19418104  0.          0.          1.          0.          1.        ]]
 
 我们可以获取每一列的特征介绍
 ['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male']
 
 可以发现其中非数值化数据变为单独的一个特征,但是这样导致整个矩阵变成稀疏矩阵，所以使用压缩矩阵存储
  (0, 0)	0.8333
  (0, 2)	1.0
  (0, 5)	1.0
 
 可以发现，这是一个三元组（数据结构的压缩矩阵表示方法）
 其中每一行中第一个元素是非0值的坐标，第二个为值
"""

decv = DictVectorizer()

x_train = decv.fit_transform(x_train)  # 先拟合，后标准化
x_test = decv.transform(x_test)

rf = RandomForestClassifier(n_estimators=10)  # max_depth=10最大树深,
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
score = rf.score(x_test, y_test)
print(score)

# 打印报告
"""
精确度: 表示的是预测为正的样本中有多少是真正的正样本。
召回率:表示的是样本中的正例有多少被预测正确了。
F1值:2*精度*召回率/(精度+召回率)
支持数: 该类在样本中出现的总次数
"""
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test,y_pred=y_pred))

from sklearn.externals import joblib  # 保存模型，下次直接就能使用无需训练

joblib.dump(rf, 'random_forest.pkl')


# In[5]:


from sklearn.externals import joblib
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.0001)
x_train = x_train.to_dict(orient='records')     # 加了orient="records"  以行操作
x_test = x_test.to_dict(orient='records')
print(len(x))
x_train = decv.fit_transform(x_train)  # 先拟合，后标准化
x_test = decv.transform(x_test)
rf_load = joblib.load('random_forest.pkl')
score = rf_load.score(x_train, y_train)
print(score)


# In[ ]:




