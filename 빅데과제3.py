ratings = spark.read.format("csv").option("sep","\t").option("inferSchema","true").load("data/ml-100k/u.data")

col_names = ["user", "item", "rating", "timestamp"]
ratingDF = ratings.toDF(*col_names)

users = spark.read.format("csv").option("sep","|").option("inferSchema","true").load("data/ml-100k/u.user")
col_uname = ["user","age","gender","occupation","zip_code"]
userDF = users.toDF(*col_uname)
userDF.show(5)
userDF.printSchema()



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

#genreDF
genre = spark.read.format("csv").option("sep","|").option("inferSchema","true").load("data/ml-100k/u.genre")
col_names = ["genre", "code"]
genreDF = genre.toDF(*col_names)

#[B]
2) a~d
ratingDF.count()
ratingDF.select("user").distinct().count()
ratingDF.select("item").distinct().count()
ratingDF.select(avg("rating"), max("rating"), min("rating")).show()
ratingDF.groupBy("rating").count().orderBy("rating").show()
ratingDF.groupBy("user").count().select(max("count"), min("count")).show()

3) a~b
userDF.select("occupation").distinct().count()
userDF.groupBy("gender").count().show()

4) a~b


5) a~b
joinDF = ratingDF.join(userDF, ratingDF.user == userDF.user, "inner")
joinDF.groupBy("gender").agg(avg("rating")).show()

ratingData = ratingDF.groupBy("item").agg(count("item"), avg("rating")).where("count(item)>=10").orderBy("avg(rating)", ascending = False).limit(5)

col_names = ["item", "count", "avg_rating"]
ratingData = ratingData.toDF(*col_names)


ratingData = ratingData.join(movieDF, ratingData.item == movieDF.item, "inner")










