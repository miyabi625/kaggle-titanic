from sklearn.ensemble import RandomForestClassifier

class Model_RandomForest:
    # constructor
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    # modelパラメータの設定
    def set_param(self,param):
        # 将来用のメソッドとして用意する。今のところは未定義（pass）
        pass

    # 機械学習
    def fit(self,x_train,y_train):
        self.model = self.model.fit(x_train, y_train)

    # 結果の取得
    def predict(self,test_data):
        return self.model.predict(test_data).astype(int)
