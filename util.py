import pandas as pd
import numpy

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
    numpy.savetxt(path, data, delimiter=',', header='userID,itemID,rating', comments='', newline='\n',
                  fmt='%0i,%0i,%10.2f')


def save_csv_from_rating(matrix, path):
    numpy.savetxt(path, matrix, delimiter=',', header='userID,itemID,rating', comments='', fmt='%0i,%0i,%0.2f')


'''
计算两个向量的Pearson 相关性
取值区间为[-1,1]。-1表示完全的负相关，+1表示完全的正相关，0表示没有线性相关。
皮尔森相关系数描述的是线性关系。严格来说，需要数据集是正态分布的，但不必是零均值的。
皮尔森相关系数的计算是先对向量每一分量减去分量均值，再求余弦相似度。这一操作称为中心化(将特征向量X根据 x¯\bar x¯x 移动)。
Pearson 相关性：两个变量的协方差除于两个变量的标准差
'''


def corrcoef(preference_matrix, u, v):
    # 求和
    x = preference_matrix[u]
    y = preference_matrix[v]
    sum1 = sum(x)
    sum2 = sum(y)
    # 求乘积之和
    sum1Sq = sum([pow(v, 2) for v in x])
    sum2Sq = sum([pow(v, 2) for v in y])

    pSum = sum([x[i] * y[i] for i in range(len(x))])

    num = pSum - (sum1 * sum2 / len(x))
    den = numpy.sqrt((sum1Sq - pow(sum1, 2) / len(x)) * (sum2Sq - pow(sum2, 2) / len(y)))

    if den == 0.0:
        return 0.0
    return num / den


'''
计算估计值与实际值的RMSE 差
rmse(numpy.array(),numpy.array())
'''


def RMSE(predictions, targets):
    return numpy.sqrt(((predictions - targets) ** 2).mean())


