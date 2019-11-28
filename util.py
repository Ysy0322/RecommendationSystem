import pandas as pd
import numpy

'''
读取csv文件
'''


def read_file(path):
    train_data = pd.read_csv(path)
    # 构建 Preference Matrix
    preference_matrix = pd.pivot_table(train_data, index='userID', columns='itemID', values=['rating'], fill_value=0.0)
    print(preference_matrix)
    preference_matrix = preference_matrix.values
    print(len(preference_matrix))
    print(len(preference_matrix[0]))
    return preference_matrix
    # result : [2967 rows x 3814 columns]


'''
计算估计值与实际值的RMSE 差
rmse(numpy.array(),numpy.array())
'''


def RMSE(predictions, targets):
    return numpy.sqrt(((predictions - targets) ** 2).mean())
