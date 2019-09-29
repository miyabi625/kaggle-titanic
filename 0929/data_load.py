import pandas as pd

class DataLoad:
    # constructor
    def __init__(self, file_path):
        # Load training data
        tmp_df = pd.read_csv(file_path, header=0)
        
        # Convert "Sex" to be a dummy variable (female = 0, Male = 1)
        tmp_df["Gender"] = tmp_df["Sex"].map({"female": 0, "male": 1}).astype(int)
        
        # Complement the missing values of "Age" column with average of "Age"
        median_age = tmp_df["Age"].dropna().median()
        if len(tmp_df.Age[tmp_df.Age.isnull()]) > 0:
            #locを用いてAgeの欠損値がある箇所に対して中央値を配置する
            tmp_df.loc[(tmp_df.Age.isnull()), "Age"] = median_age

        self.df = tmp_df

    # 該当項目の取得
    def getValues(self,param):
        return self.df[param].values
