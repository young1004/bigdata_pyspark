# evaluator def

def evaluation(rank_v, reg_v):
  als = ALS(rank = rank_v, regParam=reg_v, seed=1)
  print("rank = " + str(rank_v) + " regParam = " + str(reg_v))
  alsModel = als.fit(train)
  pred = alsModel.transform(train)
  rmse = evaluator.evaluate(pred.na.drop())
  print("train: " + str(rmse))
  pred = alsModel.transform(test)
  rmse = evaluator.evaluate(pred.na.drop())
  print("test: " + str(rmse))

for r in [10, 5, 1]:
  for reg in [1.0, 0.1, 0.01]:
    evaluation(r, reg)

userID = 1
n = 10
als = ALS(rank = 5, regParam=0.1, seed=1)
alsModel = als.fit(ratingDF)
pred = alsModel.transform(ratingDF)
user_recommend10 = alsModel.recommendForAllUsers(n)


pred.where(pred.user==1).select(pred.item).na.drop().distinct().show()
