from sklearn.model_selection import train_test_split
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction import DictVectorizer


engine = create_engine('sqlite:///MyDB.sqlite3', echo=True)
data = pd.read_sql("select * from item;", con=engine)

"""
cut将根据值本身来选择箱子均匀间隔，qcut是根据这些值的频率来选择箱子的均匀间隔。 
"""

data['num'][:] = pd.Series(pd.qcut(np.array(data['num']), 5).codes)
x = data[['name', 'level', 'hot', 'address']]
# 取目标值
y = data[['num']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# result = SelectKBest(chi2, k=2).fit_transform(x_train, y_train)
data["address"] = data["address"].apply(lambda x: x.replace("[", "").replace("]", ""))
data["province"] = data["address"].apply(lambda x: x.split("·")[0])
print(data['level'])

print(data['level'].fillna['123'])