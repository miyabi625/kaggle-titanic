####################################################
# インポート
####################################################
import data_load
import model_RandomForest
import submit_csv
import logging

####################################################
# パラメータ設定
####################################################
# ログフォーマットを定義
formatter = '%(levelname)s : %(asctime)s : %(message)s'
# ログファイル、ログレベルの設定
logging.basicConfig(filename='logfile/logger.log', level=logging.DEBUG, format=formatter)

####################################################
# データ読み込み
####################################################
logging.info('start read data')
# トレーニングデータ
train_dl = data_load.DataLoad("./input/train.csv")
# テストデータ
test_dl = data_load.DataLoad("./input/test.csv")

logging.info('end read data')

####################################################
# 分析
####################################################
logging.info('start analysis')
train_data = train_dl.getValues(["Survived","Pclass","Age","Gender","SibSp","Parch","Fare","Cabin","Embarked_NUM"])
print(train_data[1:4])

ids = test_dl.getValues(["PassengerId"])
test_data = test_dl.getValues(["Pclass","Age","Gender","SibSp","Parch","Fare","Cabin","Embarked_NUM"])

# Predict with "Random Forest"
modelRF = model_RandomForest.Model_RandomForest()
modelRF.fit(train_data[0::, 1::], train_data[0::, 0])
output = modelRF.predict(test_data)

logging.info('end analysis')

####################################################
# アウトプットファイル出力
####################################################
logging.info('start output data')
sb = submit_csv.SubmitCsv("./output/titanic_submit.csv")
sb.to_csv(["PassengerId", "Survived"],zip(ids[0::,0], output))

logging.info('end output data')
