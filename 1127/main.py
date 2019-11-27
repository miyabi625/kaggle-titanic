####################################################
# インポート
####################################################
import data_load
import model_svm as model
import submit_csv
import logger
import logging
import numpy as np

####################################################
# ログ宣言
####################################################
log = logging.getLogger(__name__)
logger.setLogger(log)

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

### トレーニングデータ用意  ###################
# 取得する項目を定義する
getList = ["Pclass","Age","Gender","Honorific","SibSp","Parch","Team","TravelAlone","SmallGroup","BigGroup","Fare","CabinRank","Embarked_NUM"]
# トレーニングデータを取得する
train_data = train_dl.getValues(getList)
train_data_Survived = train_dl.getValues(["Survived"])[0::, 0]

### パラメータ定義  ##########################
param_grid = {'C': np.logspace(-1, 2, 30)
}

### GridSearchCVインスタンス作成  ############
# Predict with "Random Forest"
modelSVM = model.Model_SVM(param_grid)

### fit  ####################################
modelSVM.fit(train_data, train_data_Survived)

log.info('feature_importances')
log.info(modelSVM.grid_search_feature_importances(getList))
log.info('best_params')
log.info(modelSVM.grid_search_best_params())
log.info('best_score')
log.info(modelSVM.grid_search_best_score())

### テストデータに適用  #######################
test_data = test_dl.getValues(getList)
ids = test_dl.getValues(["PassengerId"])
output = modelSVM.predict(test_data)

log.info('end analysis')

####################################################
# アウトプットファイル出力
####################################################
log.info('start output data')

sb = submit_csv.SubmitCsv("./output/titanic_submit.csv")
sb.to_csv(["PassengerId", "Survived"],zip(ids[0::,0], output))

log.info('end output data')
