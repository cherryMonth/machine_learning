#!/usr/bin/env python
# coding: utf-8

# In[210]:


import win32com.client
import models as ml
from sqlalchemy import Column, String, create_engine
import pandas as pd

conn=win32com.client.Dispatch('ADODB.Connection')
DSN='PROVIDER=Microsoft.Jet.OLEDB.4.0;DATA SOURCE=sjk.mdb;'  # 在此处修改数据库文件
conn.Open(DSN)
rs=win32com.client.Dispatch('ADODB.Recordset')
rs_name='co'
table_name = "testdata" # 修改此处的表名

engine = create_engine('mysql+mysqlconnector://root:123456@127.0.0.1:3306/sjk')
db_data = pd.read_sql("select * from {}".format(table_name), con=engine)
columns = tuple(db_data.columns)
length = len(columns)

rs.Open('SELECT * FROM {}'.format(table_name),conn,1,3) #1和3是常数.代表adOpenKeyset 和adLockOptimistadLockOptimistic


# In[211]:


rs.MoveFirst()
count=0
data = list()
while not rs.EOF:
    result = [None] * length
    for x in range(rs.Fields.Count):
        if str(type(rs.Fields.Item(x).Value)) == "<class 'pywintypes.datetime'>":
            result[columns.index(rs.Fields.Item(x).Name)] = str(rs.Fields.Item(x).Value)
        else:
            result[columns.index(rs.Fields.Item(x).Name)] = rs.Fields.Item(x).Value
    data.append(result)
    rs.MoveNext()


# In[213]:


data = pd.DataFrame(data, columns=tuple(db_data.columns))
data


# In[214]:


data.to_sql(table_name, con=engine, if_exists='append', index=False)


# In[ ]:




