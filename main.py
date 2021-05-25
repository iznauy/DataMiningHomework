import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, mean_squared_error

def display_dataset(df):
    # 展示数据整体信息
    print(df.info())

    # 年龄不需要修复
    ages = df['age']
    print(ages.describe())

    # 性别不需要修复
    is_males = df['is_male']
    print(is_males.describe())

    # 心率不需要修复
    heart_rates = df['Heart_Rate']
    print(heart_rates.describe())

    # 呼吸率 --> 正常值 [5, 70]，当前数据中最大值为 76，与正常数据差别不大，不需要修复
    respiratory_rate = df['Respiratory_Rate']
    print(respiratory_rate.describe())

    # 平均动脉压 --> 正常值 [10, 200]，当前最小值为 -11，最大值为 777，考虑删除掉平均动脉压在 10 以下或者在 200 以上的数据
    maps = df['MAP']
    print(maps.describe())

    # 收缩压 --> 正常值 [40, 300]，当前数据中的最小值为 0，删除舒张压在 300 以上的数据和在 40 以下的数据
    systolic_bp = df['Systolic_BP']
    print(systolic_bp.describe())

    # 舒张压 --> 正常值 [40, 300]，当前数据中的最大值为 68109，最小值为 0，应该是采集数据错误。考虑到该字段在 24 - 40 之间也有
    # 大量数据，因此删除舒张压在 300 以上的数据和在 24 以下的数据
    diastolic_bp = df['Diastolic_BP']
    print(diastolic_bp.describe())

    # 血氧饱和度 --> 正常值 [0, 100]，当前数据中的最小值为 -678。由于血氧饱和度在 50% 以下人会很快死亡，考虑剔除血氧饱和度在 50% 及以下的数据
    spo2 = df['SPO2']
    print(spo2.describe())


def drop_meaningless_data(df):
    # 考虑到训练集中有空值的数据占比不到 7%，因此选择直接剔除该部分数据（原始数据 24394 条，剔除空数据后为 22846 条）
    df.dropna(inplace=True)

    # 删除掉平均动脉压在 10 以下或者在 200 以上的数据
    df.drop(df[(df['MAP'] < 10) | (df['MAP'] > 200)].index, inplace=True)

    # 删除收缩压在 300 以上的数据和在 40 以下的数据
    df.drop(df[(df['Systolic_BP'] < 40) | (df['Systolic_BP'] > 300)].index, inplace=True)

    # 删除舒张压在 300 以上的数据和在 24 以下的数据
    df.drop(df[(df['Diastolic_BP'] < 40) | (df['Diastolic_BP'] > 300)].index, inplace=True)

    # 删除血氧饱和度在 50 以下的数据
    df.drop(df[df['SPO2'] < 50].index, inplace=True)


def model_selection():
    pass


if __name__ == '__main__':

    df = pd.read_excel('datasets/classification/train_set.xlsx')
    drop_meaningless_data(df)

    matrix = np.array(df)

    # 拆分 X 和 y
    X = matrix[:, 1:9]  # 丢弃掉无意义的第一列
    y = matrix[:, 9].astype(np.int) # 最后一列为待分类的指标

    print(np.sum(y == 0))
    print(np.sum(y == 1))
    print(np.sum(y == 2))
    print(np.sum(y == 3))
    print(np.sum(y == 4))

    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(X):
        train_X = X[train_index]
        train_y = y[train_index]
        test_X = X[test_index]
        test_y = y[test_index]
        xgb_classifier = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.7,
                                           colsample_bytree=0.7)
        xgb_classifier.fit(train_X, train_y)
        predict_y = xgb_classifier.predict(test_X)
        print(np.sum(predict_y == test_y) / len(test_y))
        print(confusion_matrix(test_y, predict_y))


    # knn = KNeighborsClassifier(n_neighbors=1)
    # knn.fit(X, y)
    #
    # predict_y = knn.predict(X[:1000])
    # print(np.sum(predict_y == y[:1000]))


