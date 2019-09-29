#---------------------------------------------------
# インポート
#---------------------------------------------------
import data_load
import model_RandomForest
import submit_csv

#---------------------------------------------------
# データ読み込み
#---------------------------------------------------
# トレーニングデータ
train_dl = data_load.DataLoad("./input/train.csv")
# テストデータ
test_dl = data_load.DataLoad("./input/test.csv")

#---------------------------------------------------
# 分析
#---------------------------------------------------
train_data = train_dl.getValues(["Survived","Pclass","Age","Gender"])

ids = test_dl.getValues(["PassengerId"])
test_data = test_dl.getValues(["Pclass","Age","Gender"])

# Predict with "Random Forest"
modelRF = model_RandomForest.Model_RandomForest()
modelRF.fit(train_data[0::, 1::], train_data[0::, 0])
output = modelRF.predict(test_data)

#---------------------------------------------------
# アウトプットファイル出力
#---------------------------------------------------
sb = submit_csv.SubmitCsv("./output/titanic_submit.csv")
sb.to_csv(["PassengerId", "Survived"],zip(ids[0::,0], output))
