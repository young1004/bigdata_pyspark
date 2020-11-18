from pyspark.sql.functions import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *
from pyspark.ml.linalg import *
from pyspark.ml.regression import *
from pyspark.ml.classification import *

# 과제 4,5,6 파일 참고

# def functions
# Logistic Regression evaluation function
def lg_eval(reg_v):
  lr = LogisticRegression(featuresCol="scaledFeatures", regParam=reg_v)
  model = lr.fit(scaledTrain)
  testSummary = model.evaluate(scaledTest)
  print("regParam = " + str(reg_v))
  print("accuracy : " + str(testSummary.accuracy))
  print("f-measure : " + str(testSummary.fMeasureByLabel(beta=1.0)[1]))
  print("areaUnderROC : " + str(testSummary.areaUnderROC))

# DecisionTree evaluation function
def dt_eval(md, mig):
  tree = DecisionTreeClassifier(maxDepth = md, minInfoGain = mig, seed = 1)
  treeModel = tree.fit(train)
  pred = treeModel.transform(test)
  evaluator = MulticlassClassificationEvaluator(metricName = "accuracy")
  accur = evaluator.evaluate(pred)
  print("maxDepth = " + str(md) + "  minInfoGain = " + str(mig) + "  accuracy = " + str(accur))

# RFormula만 바꿔서 반복하기 함수

# LinearRegression 평가함수 (RFormula 모델, 나눌 데이터프레임)
def lr_eval(RModel, data):
  transformedData = RModel.fit(data).transform(data)
  train, test = transformedData.randomSplit([0.8, 0.2], seed=1)
  scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True)
  scalerModel = scaler.fit(train)
  scaledTrain = scalerModel.transform(train)
  scaledTest = scalerModel.transform(test)
  lr = LinearRegression(featuresCol="scaledFeatures")
  lrModel = lr.fit(scaledTrain)
  print("train_r2 = ", lrModel.summary.r2)
  testSummary = lrModel.evaluate(scaledTest)
  print("test_r2 = ", testSummary.r2)

# DecisionTreeRegressor 평가함수 (RFormula 모델, 나눌 데이터프레임, metricName으로 줄 문자열 ex."r2")
def dtr_eval(RModel, data, mtName):
  transformedData = RModel.fit(data).transform(data)
  train, test = transformedData.randomSplit([0.8, 0.2], seed=1)
  dtr = DecisionTreeRegressor()
  dtrModel = dtr.fit(train)
  evaluator = RegressionEvaluator(metricName=mtName)
  trainRegress = dtrModel.transform(train)
  pred = dtrModel.transform(test)
  print("train_" + mtName + " = ", evaluator.evaluate(trainRegress))
  print("test_" + mtName + " = ", evaluator.evaluate(pred))


# 같은값만 있는 칼럼 골라내기(필요 x인 특성들 고르기)
for s in cols_web:
  if web.select(s).distinct().count() == 1:
    print(s)

# data load
web = spark.read.format("csv").option("sep", "\t").option("header", "true")\
.option("inferSchema", "true").option("nullValue", "?").load("data/webpage.tsv")
web = web.withColumn("label", col("label").cast("double"))

# data load (.fwf)
lines = spark.read.text("data/housing.data")
data= lines.select(
  lines.value.substr(1,8).cast("float").alias("CRIM"),
  lines.value.substr(9,7).cast("float").alias("ZN"),
  lines.value.substr(16,8).cast("float").alias("INDUS"),
  lines.value.substr(24,3).cast("float").alias("CHAS"),
  lines.value.substr(27,8).cast("float").alias("NOX"),
  lines.value.substr(35,8).cast("float").alias("RM"),
  lines.value.substr(43,7).cast("float").alias("AGE"),
  lines.value.substr(50,8).cast("float").alias("DIS"),
  lines.value.substr(58,4).cast("float").alias("RAD"),
  lines.value.substr(62,7).cast("float").alias("TAX"),
  lines.value.substr(69,7).cast("float").alias("PTRATIO"),
  lines.value.substr(76,7).cast("float").alias("B"),
  lines.value.substr(83,7).cast("float").alias("LSTAT"),
  lines.value.substr(90,7).cast("float").alias("MEDV"))


#Tokenizer (.fwf 2번째 방법)
rtk = RegexTokenizer(inputCol="value", outputCol="value_v")
rtkData = rtk.transform(data)
data = rtkData.select("value_v")
cols = ["value_v[" + str(x) + "]" for x in range(0, 14)]
for s in cols:
  data = data.withColumn(s, expr(s))

df = data.select(cols)

for s in cols:
  df = df.withColumn(s, col(s).cast("double"))

#헤더 없을때
col_uname = ["user","age","gender","occupation","zip_code"]
userDF = users.toDF(*col_uname)

#delete null values
web = web.na.drop("any")

# 데이터 요약 보여주기 (부족하면 과제3 참고)
web.select("alchemy_category_score", "avglinksize").describe().show()


# RFormula(feature extraction)
webFormula = RFormula(formula="label ~ . - url - urlid")
webData = webFormula.fit(web).transform(web)

# data split
train, test = webData.randomSplit([0.8, 0.2], seed = 1)
train.cache()
test.cache()

# standardization for RogisticRegression(feature transformation)
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures", min=0, max=1)
scalerModel = scaler.fit(train)

scaledTrain = scalerModel.transform(train)
scaledTest = scalerModel.transform(test)

scaledTrain.printSchema()



