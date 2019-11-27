from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import logger
import logging
import numpy as np
import pandas as pd

class Model_SVM:
    #xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)
    ####################################################
    # ログ宣言
    ####################################################
    log = logging.getLogger(__name__)
    logger.setLogger(log)
    
    # constructor
    def __init__(self,param):
        self.model = GridSearchCV(SVC(kernel='linear', random_state=None), param, cv=2, verbose=1,return_train_score=False)

    # 機械学習
    def fit(self,x_train,y_train):
        self.log.info('fit start')
        self.model.fit(x_train, y_train)

        self.log.info('fit end')

    # Best parameters
    def grid_search_feature_importances(self,getList):
        return pd.DataFrame({"feature":getList,"importance":self.model.best_estimator_.feature_importances_}).sort_values(by="importance",ascending=False)

    # Best parameters
    def grid_search_best_params(self):
        return self.model.best_params_

    # Best cross-validation
    def grid_search_best_score(self):
        return self.model.best_score_

    # 結果の取得
    def predict(self,test_data):
        return self.model.predict(test_data).astype(int)
