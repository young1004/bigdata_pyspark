from pyspark.sql.functions import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *
from pyspark.ml.linalg import *
from pyspark.ml.classification import *

def test_eval(reg_v):
  lr = LogisticRegression(featuresCol="scaledFeatures", regParam=reg_v)
  model = lr.fit(scaledTrain)
  testSummary = model.evaluate(scaledTest)
  print("regParam = " + str(reg_v))
  print("accuracy : " + str(testSummary.accuracy))
  print("f-measure : " + str(testSummary.fMeasureByLabel(beta=1.0)[1]))
  print("areaUnderROC : " + str(testSummary.areaUnderROC))

for r in [0.0, 0.01, 0.1, 1]:
  lr = LogisticRegression(featuresCol="scaledFeatures", regParam=r)
  model = lr.fit(scaledTrain)
  testSummary = model.evaluate(scaledTest)
  print("accuracy = " + str(testSummary.accuracy))
  print("f-measure = " + str(testSummary.fMeasureByLabel(beta=1.0)[1]))
  print("areaUnderROC : " + str(testSummary.areaUnderROC

for md in [1, 10, 20]:
  for mig in [0.0, 0.05]:
    tree = DecisionTreeClassifier(maxDepth = md, minInfoGain = mig, seed = 1)
    treeModel = tree.fit(train)
    pred = treeModel.transform(test)
    evaluator = MulticlassClassificationEvaluator(metricName = "accuracy")
    accur = evaluator.evaluate(pred)
    print("maxDepth = " + str(md) + "  minInfoGain = " + str(mig) + "  accuracy = " + str(accur))
