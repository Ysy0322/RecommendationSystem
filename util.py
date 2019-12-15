import pandas
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
    numpy.random.seed(1)
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


def draw_train_test(iter_array, train_rmse, test_rmse):
    plt.plot(iter_array, train_rmse, label='Train', linewidth=1)
    plt.plot(iter_array, test_rmse, label='Test', linewidth=1)

    plt.xlabel('iterations', fontsize=15)
    plt.ylabel('RMSE', fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.show()


def train_test_split(path):
    ratings, user_n, item_m = read_file(path)
    test = numpy.zeros(ratings.shape)
    train = ratings.copy()
    count = 0
    for user in range(len(user_n)):
        for item in range(len(item_m)):
            if (ratings[user][item]) != 0.0:
                count += 1
                if count % 3 == 0:
                    train[user][item] = 0
                    test[user][item] = ratings[user][item]
    # Test and train are truly disjoint
    assert (numpy.all((train * test) == 0))
    return train, numpy.array(test)


def filter_res(source_path, des_path):
    train_data = pandas.read_csv(source_path)
    train_data.drop(['userID', 'itemID'], axis=1)
    train_data['dataID'] = range(len(train_data))
    order = ['dataID', 'rating']
    train_data = train_data[order]
    numpy.savetxt(des_path, train_data, header='dataID,rating', comments='', delimiter=',', fmt='%0i,%f')
