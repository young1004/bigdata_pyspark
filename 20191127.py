# import
from pyspark.sql.functions import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *
from pyspark.ml.classification import *

# data load
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
data.show(5)
