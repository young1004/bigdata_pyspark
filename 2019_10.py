#100k movie Data

#ratingDF
rating = spark.read.format("csv").option("sep","\t").option("inferSchema","true").load("data/ml-100k/u.data")
col_names=["user","item","rating","timestamp"]
ratingDF = rating.toDF(*col_names)
ratingDF.show(5)

#userDF
user = spark.read.format("csv").option("sep","|").option("printSchema","true").load("data/ml-100k/u.user")
col_names=["user","age","gender","occupation","zip_code"]
userDF=user.toDF(*col_names)
userDF.show(5)

#movieDF
movie = spark.read.format("csv").option("sep","|").option("inferSchema","true").load("data/ml-100k/u.item")
col_names = ["_c"+str(x) for x in range(5,24)]
movieDF=movie.select(col("_c0").alias("item"),col("_c1").alias("title"),array(col_names).alias("genre_array"))


train, test = ratingDF.randomSplit([0.8, 0.2], seed=1)

from pyspark.ml.recommendation import ALS

als = ALS()

alsModel = als.fit(train)

pred = alsModel.transform(train)

alsModel.userFactors.show()

rank_v = 5
reg_v = 0.1
als = ALS(rank = rank_v, regParam = reg_v, seed=1)

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(metricName = "rmse", labelCol="rating", predictionCol="prediction")
alsModel = als.fit(train)
pred = alsModel.transform(test)
rmse = evaluator.evaluate(pred)
rmse = evaluator.evaluate(pred.na.drop())

# 2019.10.16
model = als.fit(ratingDF)
userID = 688
n = 50
movie_list = ratingDF.select("item").distinct().withColumn("user", lit(userID))
already_rated = ratingDF.where(col("user") == userID).select("item", "user")

candidate = movie_list.subtract(already_rated)
pred = model.transform(candidate).dropna().orderBy("prediction", ascending = False).select("item", "prediction").limit(n)

recomm = pred.join(movieDF, pred.item == movieDF.item, "left").select(pred.item, "prediction", "title")

mvID = 64
n = 10

user_list = ratingDF.select("user").distinct().withColumn("item", lit(mvID))
already_user = ratingDF.where(col("item") == mvID).select("user", "item")
candidate = user_list.subtract(already_user)
pred = model.transform(candidate).dropna().orderBy("prediction", ascending = False).select("user", "prediction").limit(n)


