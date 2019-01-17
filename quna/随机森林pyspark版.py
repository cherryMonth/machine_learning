sc.master

from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load("hdfs://master:9000/user/root/data/data.csv").toPandas()
data = data[data['level'] > 0]
data["area"] = data["area"].apply(lambda x: x.replace("[", "").replace("]", ""))
data["province"] = data["area"].apply(lambda x: x.split(u"·")[0])
data["city"] = data["area"].apply(lambda x: x.split(u"·")[1])

city = list(set(data['province']))
data['province'] = data['province'].apply(lambda x:city.index(x))

train_data = data[['price', 'num', 'hot', 'province', 'level']][data['level'] != None]

import seaborn as sns

num_top = data.sort_values(by='num', axis=0, ascending=False).reset_index(drop=True)
sns.set(font='SimHei')
sns.set_context("talk")
fig = plt.figure(figsize=(15, 10))
sns.barplot(num_top["name"][:30], num_top["num"][:30])
plt.xticks(rotation=90)
fig.show()

# 省份与景区评级
data['level_sum'] = 1  # 给所有行添加一列数据level_sum，值为1，用来记录总个数
var = data.groupby(['province', 'level']).level_sum.sum()
var.unstack().plot(kind='bar', figsize=(35, 10), stacked=False, color=['red', 'blue', 'green', 'yellow'])

# 人少的5A景点，4A景点，3A景点
top_5A = data[data["level"] == u"5A景区"].sort_values(by='num', axis=0, ascending=True).reset_index(drop=True)
top_4A = data[data["level"] == u"4A景区"].sort_values(by='num', axis=0, ascending=True).reset_index(drop=True)
top_3A = data[data["level"] == u"3A景区"].sort_values(by='num', axis=0, ascending=True).reset_index(drop=True)
fig = plt.figure(figsize=(15, 15))
plt.pie(top_5A["num"][:15], labels=top_5A["name"][:15], autopct='%1.2f%%')
plt.title(u"人数最少的15个5A景区")
plt.show()


from pprint import pprint
from pyspark import SparkContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint

data = sqlContext.createDataFrame(train_data)
dataSet = data.na.fill('0').rdd.map(list)
(trainData, testData) = dataSet.randomSplit([0.7, 0.3])
trainingSet = dataSet.map(lambda x:Row(label=x[-1], features=Vectors.dense(x[:-1]))).toDF()

stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
si_model = stringIndexer.fit(trainingSet)
tf = si_model.transform(trainingSet)
rf = RandomForestClassifier(numTrees=50, maxDepth=8, labelCol="indexed", seed=42)
rfcModel = rf.fit(tf)
print("模型特征重要性:{}".format(rfcModel.featureImportances))
print("模型特征数:{}".format(rfcModel.numFeatures))


testSet = testData.map(lambda x:Row(label=x[-1], features=Vectors.dense(x[:-1]))).toDF()

print("测试样本数:{}".format(testSet.count()))
print(testSet.show())

si_model = stringIndexer.fit(testSet)
test_tf = si_model.transform(testSet)

result = rfcModel.transform(test_tf)
result.show()


total_amount=result.count()
correct_amount = result.filter(result.indexed==result.prediction).count()
precision_rate = 1.0*correct_amount/total_amount
print("预测准确率为:{}".format(precision_rate))

positive_precision_amount = result.filter(result.indexed == 0).filter(result.prediction == 0).count()

negative_precision_amount = result.filter(result.indexed == 1).filter(result.prediction == 1).count()
positive_false_amount = result.filter(result.indexed == 0).filter(result.prediction == 1).count()
negative_false_amount = result.filter(result.indexed == 1).filter(result.prediction == 0).count()

print("正样本预测准确数量:{},负样本预测准确数量:{}".format(positive_precision_amount,negative_precision_amount))

positive_amount = result.filter(result.indexed == 0).count()
negative_amount = result.filter(result.indexed == 1).count()

print("正样本数:{},负样本数:{}".format(positive_amount,negative_amount))
print("正样本预测错误数量:{},负样本错误准确数量:{}".format(positive_false_amount,negative_false_amount))

recall_rate1 = positive_precision_amount/positive_amount
recall_rate2 = negative_precision_amount/negative_amount

print("正样本召回率为:{},负样本召回率为:{}".format(recall_rate1,recall_rate2))


