####################################################
# インポート
####################################################
import data_load
import model_RandomForest
import submit_csv
import logger

####################################################
# ログ宣言
####################################################
log = logger.Logger('main.py')

####################################################
# データ読み込み
####################################################
log.info('start read data')
# トレーニングデータ
train_dl = data_load.DataLoad("./input/train.csv")
# テストデータ
test_dl = data_load.DataLoad("./input/test.csv")

log.info('end read data')

####################################################
# 分析
####################################################
log.info('start analysis')
train_data = train_dl.getValues(["Survived","Pclass","Age","Gender","Honorific","SibSp","Parch","Team","Fare","CabinRank","Embarked_NUM"])
#print(train_data[1:4])

ids = test_dl.getValues(["PassengerId"])
test_data = test_dl.getValues(["Pclass","Age","Gender","Honorific","SibSp","Parch","Team","Fare","CabinRank","Embarked_NUM"])

# Predict with "Random Forest"
modelRF = model_RandomForest.Model_RandomForest()
modelRF.fit(train_data[0::, 1::], train_data[0::, 0])
output = modelRF.predict(test_data)

log.info('end analysis')

####################################################
# アウトプットファイル出力
####################################################
log.info('start output data')
sb = submit_csv.SubmitCsv("./output/titanic_submit.csv")
sb.to_csv(["PassengerId", "Survived"],zip(ids[0::,0], output))

log.info('end output data')
