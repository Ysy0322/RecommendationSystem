import pandas as pd
import numpy

'''
读取csv文件
获取preference matrix
'''


def read_file(path):
    train_data = pd.read_csv(path)
    # 构建 Preference Matrix
    preference_matrix = pd.pivot_table(train_data, index='userID', columns='itemID', values=['rating'], fill_value=0.0)
    preference_matrix = preference_matrix.values
    # result : [2967 rows x 3814 columns]
    return preference_matrix


'''
preference matrix写入csv文件
'''

'''
计算估计值与实际值的RMSE 差
rmse(numpy.array(),numpy.array())
'''


def RMSE(predictions, targets):
    return numpy.sqrt(((predictions - targets) ** 2).mean())
