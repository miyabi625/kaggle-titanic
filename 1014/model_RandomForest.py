from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import logger
import logging
import numpy as np

class Model_RandomForest:
    ####################################################
    # ログ宣言
    ####################################################
    log = logging.getLogger(__name__)
    logger.setLogger(log)
    
    # constructor
    def __init__(self,param):
        self.model = GridSearchCV(RandomForestClassifier(random_state=0), param, cv=2,return_train_score=False)

    # 機械学習
    def fit(self,x_train,y_train):
        self.log.info('fit start')
        self.model.fit(x_train, y_train)

        self.log.info('fit end')

    # Best parameters
    def grid_search_feature_importances(self):
        return self.model.best_estimator_.feature_importances_

    # Best parameters
    def grid_search_best_params(self):
        return self.model.best_params_

    # Best cross-validation
    def grid_search_best_score(self):
        return self.model.best_score_

    # 結果の取得
    def predict(self,test_data):
        return self.model.predict(test_data).astype(int)
