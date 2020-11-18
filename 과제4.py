# import
from pyspark.sql.functions import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *
from pyspark.ml.classification import *

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

# data head (ubuntu shell)
head -2 data/webpage.tsv

# data load
web = spark.read.format("csv").option("sep", "\t").option("header", "true")\
.option("inferSchema", "true").option("nullValue", "?").load("data/webpage.tsv")
web = web.withColumn("label", col("label").cast("double"))

#delete null values
web = web.na.drop("any")

#columns value duplicate
cols_web = web.columns[2:25]
for s in cols_web:
  if web.select(s).distinct().count() == 1:
    print(s)

# drop framebased, is_news
web.drop("framebased")
web = web.drop("is_news")

# data analysis
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
scaledTrain.select("scaledFeatures").show(truncate=False)

#Logistic Regression & evaluate
for r in [0.0, 0.01, 0.1, 1]:
  lg_eval(r)

# regPram 0.1

# Decision Tree evaluate
for md in [1, 10, 20]:
  for mig in [0.0, 0.05]:
    dt_eval(md, mig)

#maxDepth=10, minInfoGain=0.0 seed=1
tree = DecisionTreeClassifier(maxDepth=10, seed=1)
treeModel = tree.fit(train)
pred=treeModel.transform(test)
evaluator = MulticlassClassificationEvaluator() 
evaluator.evaluate(pred) # f1 score(f-measure)
evaluator.setMetricName("accuracy").evaluate(pred) #accuracy

confusion_matrix = pred.groupBy("label").pivot("prediction", [0,1]).count().na.fill(0.0)\
.orderBy("label")
confusion_matrix.show()



