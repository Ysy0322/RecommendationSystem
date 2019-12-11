import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn

'''
读取csv文件
获取preference matrix
'''


def read_file(path, train=True):
    if train:
        train_data = pd.read_csv(path)
        # result : [2967 rows x 3814 columns]
        # item的index与item的id不是对应的
        user = train_data['userID']
        item = train_data['itemID']
        user = list(set(user))
        item = list(set(item))

        # 构建 Preference Matrix
        preference_matrix = pd.pivot_table(train_data, index='userID', columns='itemID', values=['rating'],
                                           fill_value=0.0)
        preference_matrix = preference_matrix.values
        return preference_matrix, user, item
    else:
        df = pd.read_csv(path)
        # for self_test
        # df = df.drop(['rating', 'timestamp'], axis=1)
        return df.values


'''
preference matrix写入csv文件
'''


def save_csv_from_preference_matrix(matrix, path):
    data = pd.DataFrame(matrix)
    data = data.stack()
    data.index = data.index.rename('userID', level=0)
    data.index = data.index.rename('itemID', level=1)
    data.name = 'rating'
    data = data.reset_index()
    numpy.savetxt(path, data, delimiter=',', header='userID,itemID,rating', comments='', newline='\n')


'''
userID,itemID,rating 写入csv文件
'''


def save_csv_from_rating(matrix, path):
    numpy.savetxt(path, matrix, delimiter=',', header='userID,itemID,rating', comments='', fmt='%0i,%0i,%f')


'''
生成随机数矩阵
0~1的随机浮点数
'''


def generate_random(row, col):
    numpy.random.seed(0)
    return numpy.random.random((row, col))


'''
计算预测过程中的mean_squared_error
'''


def get_rmse(predict, target):
    predict = predict[target.nonzero()].flatten()
    target = target[target.nonzero()].flatten()
    return numpy.sqrt(((predict - target) ** 2).mean())


'''
for self_test
计算估计值与实际值的get_rmse 差
get_rmse(numpy.array(),numpy.array())
'''


def get_rmse_with_csv(predict_path, target_path):
    predict = pd.read_csv(predict_path)
    target = pd.read_csv(target_path)

    rating1 = numpy.array(predict['rating'])
    rating2 = numpy.array(target['rating'])

    return numpy.sqrt(((rating1 - rating2) ** 2).mean())


'''
画图
'''


def draw(iter_array, rmse_array):
    seaborn.set()
    color = ['g', 'r', 'b', 'y', 'c']
    # k w
    for i in range(len(rmse_array)):
        plt.plot(iter_array, rmse_array[i][1], label='Train for k_factors = ' + str(rmse_array[i][0]), linewidth=1,
                 color=color[i])
    plt.xlabel('iterate times', fontsize=15)
    plt.ylabel('RMSE', fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.show()
