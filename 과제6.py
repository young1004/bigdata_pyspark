from pyspark.sql.functions import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *
from pyspark.ml.linalg import *
from pyspark.ml.regression import *
from pyspark.ml.classification import *

head -2 bike.csv

#data load
data = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load("data/bike.csv")
cols_data = data.columns[8:14]
data = data.select("season", "holiday", *cols_data, "cnt" )
data = data.withColumn("cnt", col("cnt").cast("double"))

# 데이터 분석 #1), 2), 3)
data.groupBy("season").agg(avg("cnt")).orderBy("season").show()
data.groupBy("weathersit").agg(avg("cnt")).orderBy("weathersit").show()
data.select(corr("temp", "cnt"), corr("atemp","cnt"), corr("hum", "cnt"), corr("windspeed", "cnt")).show()

#RFormula 4)~ 8)
RFModel = RFormula(formula="cnt ~ temp") # 4) a
RFModel = RFormula(formula="cnt ~ temp+atemp") # 4) b
RFModel = RFormula(formula="cnt ~ temp+hum") # 4) c
RFModel = RFormula(formula="cnt ~ temp+atemp+hum+windspeed") # 4) d

RFModel = RFormula(formula="cnt ~ season") # 5)

# 6)
onehot = OneHotEncoderEstimator(inputCols = ["season"], outputCols = ["season_onehot"]) 
data = onehot.fit(data).transform(data)
RFModel = RFormula(formula="cnt ~ season_onehot")
# 6) 끝

RFModel = RFormula(formula="cnt ~ .-season_onehot") # 7)

# 8)
onehot = OneHotEncoderEstimator(inputCols = ["holiday", "workingday", "weathersit"], outputCols = ["holiday_onehot", "workingday_onehot", "weathersit_onehot"])
data = onehot.fit(data).transform(data)
cols_data = data.columns[4:]
data = data.select(*cols_data)
RFModel = RFormula(formula="cnt ~ .")
# 8) 끝

# 9) 
RFModel = RFormula(formula="cnt ~ .") #9)-1

# 9)-2
onehot = OneHotEncoderEstimator(inputCols = ["season", "holiday", "workingday", "weathersit"], outputCols = ["season_onehot", "holiday_onehot", "workingday_onehot", "weathersit_onehot"])
data = onehot.fit(data).transform(data)
cols_data = data.columns[4:]
data = data.select(*cols_data)
RFModel = RFormula(formula="cnt ~ .")
# 9) 끝

# 여기부터 반복
transformedData = RFModel.fit(data).transform(data)

#data split
train, test = transformedData.randomSplit([0.8, 0.2], seed=1)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True)
scalerModel = scaler.fit(train)
scaledTrain = scalerModel.transform(train)
scaledTest = scalerModel.transform(test)


# 4)~8) 이어서
lr = LinearRegression(featuresCol="scaledFeatures")
lrModel = lr.fit(scaledTrain)
lrModel.summary.r2

testSummary = lrModel.evaluate(scaledTest)
testSummary.r2

# 9)
dtr = DecisionTreeRegressor()
dtrModel = dtr.fit(train)
evaluator = RegressionEvaluator(metricName="r2")

trainRegress = dtrModel.transform(train)
pred = dtrModel.transform(test)

evaluator.evaluate(trainRegress)
evaluator.evaluate(pred)


