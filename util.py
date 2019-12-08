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
    numpy.savetxt(path, matrix, delimiter=',', header='userID,itemID,rating', comments='', fmt='%0i,%0i,%0.1f')


'''
计算估计值与实际值的RMSE 差
rmse(numpy.array(),numpy.array())
'''


def RMSE(predict1_path, predict2_path):
    predict1 = pd.read_csv(predict1_path)
    predict2 = pd.read_csv(predict2_path)

    rating1 = numpy.array(predict1['rating'])
    rating2 = numpy.array(predict2['rating'])

    return numpy.sqrt(((rating1 - rating2) ** 2).mean())


rmse_3_5 = RMSE("data\\user_based\\user_based_predict_k_3.csv", "data\\user_based\\user_based_predict_k_5.csv")
rmse_3_10 = RMSE("data\\user_based\\user_based_predict_k_3.csv", "data\\user_based\\user_based_predict_k_10.csv")
rmse_5_10 = RMSE("data\\user_based\\user_based_predict_k_5.csv", "data\\user_based\\user_based_predict_k_10.csv")
rmse_5_20 = RMSE("data\\user_based\\user_based_predict_k_5.csv", "data\\user_based\\user_based_predict_k_20.csv")
rmse_10_20 = RMSE("data\\user_based\\user_based_predict_k_10.csv", "data\\user_based\\user_based_predict_k_20.csv")
rmse_10_50 = RMSE("data\\user_based\\user_based_predict_k_10.csv", "data\\user_based\\user_based_predict_k_50.csv")
rmse_10_all = RMSE("data\\user_based\\user_based_predict_k_10.csv",
                   "data\\user_based\\user_based_predict_without_k.csv")

print("RMSE between k=3 and k=5 is: " + str(rmse_3_5))
print("RMSE between k=3 and k=10 is: " + str(rmse_3_10))
print("RMSE between k=5 and k=10 is: " + str(rmse_5_10))
print("RMSE between k=5 and k=20 is: " + str(rmse_5_20))
print("RMSE between k=10 and k=20 is: " + str(rmse_10_20))
print("RMSE between k=10 and k=50 is: " + str(rmse_10_50))
print("RMSE between k=10 and without k is: " + str(rmse_10_all))
