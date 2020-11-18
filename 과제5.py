from pyspark.sql.functions import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *
from pyspark.ml.classification import *

# linux shell head
head -5 data/aaa.data

# data load
data = spark.read.text("data/aaa.data")

#Tokenizer
rtk = RegexTokenizer(inputCol="value", outputCol="value_v")
rtkData = rtk.transform(data)
data = rtkData.select("value_v")
cols = ["value_v[" + str(x) + "]" for x in range(0, 14)]
for s in cols:
  data = data.withColumn(s, expr(s))

df = data.select(cols)

for s in cols:
  df = df.withColumn(s, col(s).cast("double"))


