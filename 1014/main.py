####################################################
# インポート
####################################################
import data_load
import model_RandomForest as model
import submit_csv
import logger
import logging
from tqdm import tqdm

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
getList = ["Pclass","Age","Gender","Honorific","SibSp","Parch","Team","Fare","CabinRank","Embarked_NUM"]
# トレーニングデータを取得する
train_data = train_dl.getValues(getList)
train_data_Survived = train_dl.getValues(["Survived"])[0::, 0]

### パラメータ定義  ##########################
#param_grid = {"max_depth": [2,3, None],
#      "n_estimators":[50,100,200,300,400,500],
#      "max_features": [1, 3, 5, 7, 10],
#      "min_samples_split": [2, 3, 10],
#      "min_samples_leaf": [1, 3, 10],
#      "bootstrap": [True, False],
#      "criterion": ["gini", "entropy"]}

param_grid = {'max_depth': [1, 5, 10, None],
    'n_estimators': [100],
    'max_features': [1, 'auto', None],
    'min_samples_leaf': [1, 2, 4,]
}

### GridSearchCVインスタンス作成  ############
# Predict with "Random Forest"
modelRF = model.Model_RandomForest(param_grid)

### fit  ####################################
tqdm(modelRF.fit(train_data, train_data_Survived))

log.info('feature_importances')
log.info(modelRF.grid_search_feature_importances())
log.info('best_params')
log.info(modelRF.grid_search_best_params())
log.info('best_score')
log.info(modelRF.grid_search_best_score())

### テストデータに適用  #######################
test_data = test_dl.getValues(getList)
ids = test_dl.getValues(["PassengerId"])
output = modelRF.predict(test_data)

log.info('end analysis')

####################################################
# アウトプットファイル出力
####################################################
log.info('start output data')

sb = submit_csv.SubmitCsv("./output/titanic_submit.csv")
sb.to_csv(["PassengerId", "Survived"],zip(ids[0::,0], output))

log.info('end output data')
