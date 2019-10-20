from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import logger
import numpy as np

class Model_RandomForest:
    ####################################################
    # ログ宣言
    ####################################################
    log = logger.Logger('model_RandomForest.py')

    # constructor
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    # modelパラメータの設定
    def set_param(self,param):
        # 将来用のメソッドとして用意する。今のところは未定義（pass）
        pass

    # 機械学習
    def fit(self,x_train,y_train):
        self.log.info('fit start')
        self.model = self.model.fit(x_train, y_train)

        self.log.info('fit end')

    # 特徴量の重要度の測定
    def feature_importances(self,arrlist,x_train, y_train):
        self.log.debug('feature_importances start')

        self.model.fit(x_train, y_train)

        # Feature Importance
        self.log.debug('Feature Importance:')
        fti = self.model.feature_importances_  
        for i, arrval in enumerate(arrlist):
            self.log.debug('\t{0:20s} : {1:>.6f}'.format(arrval, fti[i]))

        self.log.debug('feature_importances end')

        return fti

    # 分類結果の評価
    def sk_fold(self,x_train, y_train):
        self.log.debug('sk_fold start')

        # 層化 k 分割交差検証
        stratifiedkfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
        scores = cross_val_score(self.model, x_train, y_train, cv=stratifiedkfold)
        self.log.debug('stratifiedkfold scores = {} : {}'.format(np.mean(scores),scores))

        self.log.debug('sk_fold end')

        return np.mean(scores)

    # 結果の取得
    def predict(self,test_data):
        return self.model.predict(test_data).astype(int)
