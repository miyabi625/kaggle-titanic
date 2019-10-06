import pandas as pd
import numpy as np
import logging

class DataLoad:
    #定数宣言
    CABIN_FARE_SAMPLING = 10 #１Cabin当たりの料金の刻み幅
    
    # constructor
    def __init__(self, file_path):
        # Load training data
        tmp_df = pd.read_csv(file_path, header=0)

        #データ編集、データ補完
        # Sex（Gender）
        # Convert "Sex" to be a dummy variable (female = 0, Male = 1)
        tmp_df["Gender"] = tmp_df["Sex"].map({"female": 0, "male": 1}).astype(int)

        # honorific
        #名前に敬称が付いており、生存率に影響すると思われるため、敬称項目を追加する
        tmp_df.loc[tmp_df["Name"].str.contains("Mr."), "Honorific"] = 1
        tmp_df.loc[tmp_df["Name"].str.contains("Miss."), "Honorific"] = 2
        tmp_df.loc[tmp_df["Name"].str.contains("Mrs."), "Honorific"] = 3
        tmp_df.loc[tmp_df["Name"].str.contains("Master."), "Honorific"] = 4
        tmp_df.loc[tmp_df["Name"].str.contains("Dr."), "Honorific"] = 5
        tmp_df.loc[tmp_df["Name"].str.contains("Rev."), "Honorific"] = 6
        tmp_df.loc[tmp_df["Name"].str.contains("Col."), "Honorific"] = 7
        tmp_df.loc[tmp_df["Name"].str.contains("Major."), "Honorific"] = 8
        tmp_df.loc[tmp_df["Name"].str.contains("Mlle."), "Honorific"] = 9
        tmp_df.loc[tmp_df["Name"].str.contains("Capt."), "Honorific"] = 10
        tmp_df.loc[tmp_df["Name"].str.contains("Don."), "Honorific"] = 11
        tmp_df.loc[tmp_df["Name"].str.contains("Jonkheer."), "Honorific"] = 12
        tmp_df.loc[tmp_df["Name"].str.contains("Lady."), "Honorific"] = 13
        tmp_df.loc[tmp_df["Name"].str.contains("Mme."), "Honorific"] = 14
        tmp_df.loc[tmp_df["Name"].str.contains("Ms."), "Honorific"] = 15
        tmp_df.loc[tmp_df["Name"].str.contains("Sir."), "Honorific"] = 16
        tmp_df.loc[tmp_df["Name"].str.contains("Countess."), "Honorific"] = 17
        tmp_df.loc[tmp_df.Honorific.isnull(), "Honorific"] = 0

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

        # Ticket数のカウント列
        for TicketValue in set(tmp_df["Ticket"].values):
            TicketCnt = (tmp_df["Ticket"] == TicketValue).sum()
            tmp_df.loc[(tmp_df["Ticket"] == TicketValue), "TicketCnt"] = TicketCnt
        
        # Team
        #チケット数と同乗者数が一致しないものがあるので、友人などデータにない情報があると思われる
        #家族や友人などの仲間がいると協力プレイで生存率が高まると考えられるため、仲間（チーム）の人数項目を追加する
        
        # SibSpとParchには自分が含まれていないので＋１する。チケット数と比較して大きい方をチームの人数とする
        tmp_df["Team"] = (tmp_df["SibSp"] + tmp_df["Parch"] + 1) 
        tmp_df.loc[tmp_df["Team"] < tmp_df["TicketCnt"], "Team"] = tmp_df["TicketCnt"]
        
        # Fare（料金）
        #料金がチケット数の合計（合算）になっているようなので、１人あたりの料金に割り戻す
        tmp_df["Fare"] = tmp_df["Fare"] / tmp_df["TicketCnt"]
        #料金＝0も欠損値として扱う
        tmp_df.loc[tmp_df["Fare"] == 0, "Fare"] = None
        #料金を割り戻した後に、クラスチケット毎の中央値を求める
        median_fare = tmp_df["Fare"].dropna().median()
        median_fare1 = tmp_df[tmp_df["Pclass"] == 1]["Fare"].dropna().median()
        median_fare2 = tmp_df[tmp_df["Pclass"] == 2]["Fare"].dropna().median()
        median_fare3 = tmp_df[tmp_df["Pclass"] == 3]["Fare"].dropna().median()
        logging.debug("Fare median_fare=%s median_fare1=%s median_fare2=%s median_fare3=%s",median_fare,median_fare1,median_fare2,median_fare3)
        
        if len(tmp_df.Fare[tmp_df.Fare.isnull()]) > 0:
            #locを用いてFareの欠損値がある箇所に対して中央値を配置する
            tmp_df.loc[(tmp_df.Fare.isnull())&(tmp_df["Pclass"] == 1), "Fare"] = median_fare1
            tmp_df.loc[(tmp_df.Fare.isnull())&(tmp_df["Pclass"] == 2), "Fare"] = median_fare2
            tmp_df.loc[(tmp_df.Fare.isnull())&(tmp_df["Pclass"] == 3), "Fare"] = median_fare3
        
        #Cabin項目を見ると、１チケットで複数のCabinを取っているケースがあるため、１Cabinあたりの料金も求めておく
        tmp_df["CabinCnt"] = tmp_df["Cabin"].str.count(" ")+1
        tmp_df.loc[(tmp_df.Cabin.isnull()), "CabinCnt"] = 1
        #CABIN_FARE_SAMPLINGで設定した刻み幅で1Cabin当たりの料金を保持する
        tmp_df["CabinFare"] = self.CABIN_FARE_SAMPLING * (((tmp_df["Fare"] / tmp_df["CabinCnt"]) // self.CABIN_FARE_SAMPLING) + 1)
        
        # Cabin（客室番号）
        #Cabinは、A*：0、B*：1、C*：2、D*：3、E*：4、F*：5、G*：6、T*：7とする（複数存在時は上位層に合わせる）
        tmp_df.loc[(tmp_df.Cabin.notnull())&(tmp_df["Cabin"].str.contains("A")), "CabinRank"] = 0
        tmp_df.loc[(tmp_df.Cabin.notnull())&(tmp_df["Cabin"].str.contains("B")), "CabinRank"] = 1
        tmp_df.loc[(tmp_df.Cabin.notnull())&(tmp_df["Cabin"].str.contains("C")), "CabinRank"] = 2
        tmp_df.loc[(tmp_df.Cabin.notnull())&(tmp_df["Cabin"].str.contains("D")), "CabinRank"] = 3
        tmp_df.loc[(tmp_df.Cabin.notnull())&(tmp_df["Cabin"].str.contains("E")), "CabinRank"] = 4
        tmp_df.loc[(tmp_df.Cabin.notnull())&(tmp_df["Cabin"].str.contains("F")), "CabinRank"] = 5
        tmp_df.loc[(tmp_df.Cabin.notnull())&(tmp_df["Cabin"].str.contains("G")), "CabinRank"] = 6
        tmp_df.loc[(tmp_df.Cabin.notnull())&(tmp_df["Cabin"].str.contains("T")), "CabinRank"] = 7
        
        #Cabinは欠損値が多いため、いくつかの情報を持ち合わせて推測する
        before_median_CabinRank = 8 #一旦8で初期化する
        for CabinFareValue in sorted(set(tmp_df["CabinFare"].values)):
            #1Cabin当たりの料金に対してのCabinRankの中央値を求める
            median_CabinRank = tmp_df[tmp_df["CabinFare"] == CabinFareValue]["CabinRank"].dropna().median()

            # 取得不可（null値）の場合は一つ前のレベルで代用する
            if np.isnan(median_CabinRank):
                median_CabinRank = before_median_CabinRank
            
            #中央値をセットする
            tmp_df.loc[(tmp_df.CabinRank.isnull())&(tmp_df["CabinFare"] == CabinFareValue), "CabinRank"] = median_CabinRank
            logging.debug("median_CabinRank CabinFareValue=%s median_CabinRank=%s",CabinFareValue,median_CabinRank)
            #一つ前のレベルをキープする
            before_median_CabinRank = median_CabinRank

        # Embarked（出向地）
        #欠損値はデータ量の多い「S」とする
        #Embarkedは、S：0、C：1、Q：2とする
        tmp_df.loc[(tmp_df.Embarked.isnull()), "Embarked"] = "S"
        tmp_df["Embarked_NUM"] = tmp_df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)


        self.df = tmp_df

    # 該当項目の取得
    def getValues(self,param):
        return self.df[param].values
