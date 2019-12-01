import pandas
import pandas as pd
import numpy
import user_based

'''
读取csv文件
获取preference matrix
'''


def read_file(path):
    train_data = pd.read_csv(path)
    # 构建 Preference Matrix
    preference_matrix = pd.pivot_table(train_data, index='userID', columns='itemID', values=['rating'], fill_value=0.0)
    preference_matrix = preference_matrix.values
    print(preference_matrix)
    # result : [2967 rows x 3814 columns]
    return preference_matrix


'''
preference matrix写入csv文件
'''


def save_csv(matrix, path):
    data = pandas.DataFrame(matrix)
    data = data.stack()
    data.index = data.index.rename('userID', level=0)
    data.index = data.index.rename('itemID', level=1)
    data.name = 'rating'
    data = data.reset_index()
    numpy.savetxt(path, data, delimiter=',', header='userID,itemID,rating', comments='', newline='\n',
                  fmt='%0i,%0i,%10.7f')


'''
计算估计值与实际值的RMSE 差
rmse(numpy.array(),numpy.array())
'''


def RMSE(predictions, targets):
    return numpy.sqrt(((predictions - targets) ** 2).mean())


'''
方法一：写入csv文件，比较文件
问题：写入之后还是要读出来？所以写入就没有必要了
方法二：读取实际评分的csv文件，获取preference matrix
               与预测的csv文件比较
方法二比较可取 √
'''


def test_RMSE():
    UB = user_based.userBasedRecommSys()
    UB.setup("train_mini.csv", 3)
    UB.predicate_rating()
    print(RMSE(UB.preference_matrix, read_file("test_mini.csv")))


if __name__ == '__main__':
    test_RMSE()
