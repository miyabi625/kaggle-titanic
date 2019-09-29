# https://qiita.com/taka4sato/items/802c494fdebeaa7f43b7

import pandas as pd
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# Load training data
#header=0はheaderが行トップにあるときの記述法
train_df = pd.read_csv("./input/train.csv", header=0)

# Convert "Sex" to be a dummy variable (female = 0, Male = 1)
#train_dfにgenderw列を追加し、femaleは0,maleは1として入力
train_df["Gender"] = train_df["Sex"].map({"female": 0, "male": 1}).astype(int)
#train_df.head(3)

# Complement the missing values of "Age" column with average of "Age"
#Ageの欠損値を取り除いたものから中央値を求める。
median_age = train_df["Age"].dropna().median()
if len(train_df.Age[train_df.Age.isnull()]) > 0:
  #locを用いてAgeの欠損値がある箇所に対して中央値を配置する
  train_df.loc[(train_df.Age.isnull()), "Age"] = median_age

# remove un-used columns
#列を指定してその行を削除
train_df = train_df.drop(["Name", "Ticket", "Sex", "SibSp", "Parch", "Fare", "Cabin",
 "Embarked", "PassengerId"], axis=1)
#上から三行を入力？
#train_df.head(3)

# Load test data, Convert "Sex" to be a dummy variable
#header=0はheaderが行トップにあるときの記述法。テストデータ読み込み
test_df = pd.read_csv("./input/test.csv", header=0)
#test_dfにgenderw列を追加し、femaleは0,maleは1として入力
test_df["Gender"] = test_df["Sex"].map({"female": 0, "male": 1}).astype(int)

print(test_df.head(4))

# Complement the missing values of "Age" column with average of "Age"
#Ageの欠損値を取り除いたものから中央値を求める。
median_age = test_df["Age"].dropna().median()
if len(test_df.Age[test_df.Age.isnull()]) > 0:
  #locを用いてAgeの欠損値がある箇所に対して中央値を配置する
  test_df.loc[(test_df.Age.isnull()), "Age"] = median_age

# Copy test data's "PassengerId" column, and remove un-used columns
#idsにPassengerIdをコピー
ids = test_df["PassengerId"].values
#列を指定してその行を削除
test_df = test_df.drop(["Name", "Ticket", "Sex", "SibSp", "Parch", "Fare", "Cabin", 
"Embarked", "PassengerId"], axis=1)
#test_df.head(3)

# Predict with "Random Forest"
train_data = train_df.values
test_data = test_df.values
model = RandomForestClassifier(n_estimators=100)
output = model.fit(train_data[0::, 1::], train_data[0::, 0]).predict(test_data).astype(int)

# export result to be "titanic_submit.csv"
submit_file = open("./output/titanic_submit.csv", "w", newline="")
file_object = csv.writer(submit_file)
file_object.writerow(["PassengerId", "Survived"])
file_object.writerows(zip(ids, output))
submit_file.close()
