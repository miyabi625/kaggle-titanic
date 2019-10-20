####################################################
# インポート
####################################################
import data_load
import model_RandomForest as model
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

# 取得する項目を定義する
getList = ["Pclass","Age","Gender","Honorific","SibSp","Parch","Team","Fare","CabinRank","Embarked_NUM"]

# トレーニングデータを取得する
train_data = train_dl.getValues(getList)
train_data_Survived = train_dl.getValues(["Survived"])[0::, 0]

# Predict with "Random Forest"
modelRF = model.Model_RandomForest()
# 特徴量の重要度を測定する
fti = modelRF.feature_importances(getList,train_data, train_data_Survived)

# 特徴量の重要度の低い順から項目を削除し、最もスコアの高いリストを求める
score = 0
threshold = 0.00
# 一つ前をキープ
getList_fti_bef = []
for i in range(11):
    getList_fti = getList[:]
    for t, arrval in enumerate(getList):
        if fti[t] < threshold:
            getList_fti.remove(arrval)

    if getList_fti == getList_fti_bef:
        # 変更がない場合は後続処理をスキップする
        threshold += 0.01
        continue

    train_data = train_dl.getValues(getList_fti)
    
    log.debug("list={}".format(getList_fti))
    score_wk = modelRF.sk_fold(train_data, train_data_Survived)

    if score < score_wk:
        getList_fti_best = getList_fti
        score = score_wk
        log.info("score={} list={}".format(score,getList_fti_best))

    getList_fti_bef = getList_fti[:]
    threshold += 0.01

train_data = train_dl.getValues(getList_fti_best)
modelRF.fit(train_data, train_data_Survived)

# テストデータに適用
log.debug("list={}".format(getList_fti_best))
test_data = test_dl.getValues(getList_fti_best)
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
