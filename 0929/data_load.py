import pandas as pd
import logging

class DataLoad:
    # constructor
    def __init__(self, file_path):
        # Load training data
        tmp_df = pd.read_csv(file_path, header=0)

        #データ編集、データ補完
        #Sex（Gender）
        # Convert "Sex" to be a dummy variable (female = 0, Male = 1)
        tmp_df["Gender"] = tmp_df["Sex"].map({"female": 0, "male": 1}).astype(int)
        
        # Age
        #チケットクラスの購入年層は異なると思われるため、チケットクラス毎に中央値を算出する
        median_age1 = tmp_df[tmp_df["Pclass"] == 1]["Age"].dropna().median()
        median_age2 = tmp_df[tmp_df["Pclass"] == 2]["Age"].dropna().median()
        median_age3 = tmp_df[tmp_df["Pclass"] == 3]["Age"].dropna().median()
        logging.debug("Age median_age1=%s median_age2=%s median_age3=%s",median_age1,median_age2,median_age3)

        if len(tmp_df.Age[tmp_df.Age.isnull()]) > 0:
            #locを用いてAgeの欠損値がある箇所に対して中央値を配置する
            tmp_df.loc[(tmp_df.Age.isnull())&(tmp_df["Pclass"] == 1), "Age"] = median_age1
            tmp_df.loc[(tmp_df.Age.isnull())&(tmp_df["Pclass"] == 2), "Age"] = median_age2
            tmp_df.loc[(tmp_df.Age.isnull())&(tmp_df["Pclass"] == 3), "Age"] = median_age3
        
        # Parch（同乗の親/子供の数）
        #この項目には乳母が含まれていないとの事なので、１人では乗らないであろう15歳以下の0を1に変更する
        tmp_df.loc[(tmp_df["Age"] <= 15)&(tmp_df["Parch"] == 0), "Parch"] = 1

        # Fare（料金）
        median_fare = tmp_df["Fare"].dropna().median()
        if len(tmp_df.Fare[tmp_df.Fare.isnull()]) > 0:
            #locを用いてFareの欠損値がある箇所に対して中央値を配置する
            tmp_df.loc[(tmp_df.Fare.isnull()), "Fare"] = median_fare

        # Cabin（客室番号）
        #Cabinが存在する場合は一律「1」とする
        tmp_df.loc[(tmp_df.Cabin.notnull()), "Cabin"] = 1
        #Cabinの欠損値は「0」をセットする
        tmp_df.loc[(tmp_df.Cabin.isnull()), "Cabin"] = 0

        # Embarked（出向地）
        #欠損値はデータ量の多い「S」とする
        #Embarkedは、S：0、C：1、Q：2とする
        tmp_df.loc[(tmp_df.Embarked.isnull()), "Embarked"] = "S"
        tmp_df["Embarked_NUM"] = tmp_df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)


        self.df = tmp_df

    # 該当項目の取得
    def getValues(self,param):
        return self.df[param].values
