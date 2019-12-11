"""
Based on ALS 交替最小二乘法(alternative least square)
先固定 X 优化 Y ，然后固定 Y 优化 X，不断重复，直到 X 和 Y 收敛为止
每次固定其中一个优化另一个都需要解一个最小二乘问题
"""
import datetime
import numpy as np
from numpy.linalg import solve
import util


class PMF_ALS:
    def __init__(self):
        self.preference_matrix = []
        self.user_k_n = []
        self.item_k_m = []
        self.user = []
        self.item = []
        self.user_n = 0
        self.item_m = 0
        self.k_factors = 0

    '''
    初始化用户隐藏因子矩阵和商品隐藏因子矩阵
    user_n_k取值范围0-1的n*k随机矩阵
    item_m_k取值范围0-1的m*k随机矩阵
    '''

    def setup(self, path, k_factors):
        self.preference_matrix, self.user, self.item = util.read_file(path)
        self.user_n = len(self.user)
        self.item_m = len(self.item)
        self.k_factors = k_factors
        self.user_k_n = util.generate_random(self.user_n, self.k_factors)
        self.item_k_m = util.generate_random(self.item_m, self.k_factors)

    '''
    训练用户矩阵：
    固定商品m*k_factors的矩阵，让它的值不变，求出损失函数最小值的用户矩阵。
    算法: 损失函数 mse 对U求偏导，令导数=0，求解矩阵
    ∑(i，m)(IiT Ii + λE) Un = ∑(i, m)Ri，j Ii  
    Un = （ITI + λE）-1 Ri I   
    训练商品矩阵：
    固定用户n*k_factors的矩阵，让它的值不变，求出损失函数最小值的商品矩阵。  
    算法: 损失函数 mse 对M求偏导，令导数=0，求解矩阵
    ∑(j，n)(UjT Uj + λE) Im = ∑(j, m)Ri, j Uj
    Im = （UTU + λE）-1 RjT U  
    '''

    def als_function(self, latent_matrix, fixed_matrix, ratings, reg, type='user'):
        FTF = fixed_matrix.T.dot(fixed_matrix)
        lambdaE = np.eye(self.k_factors) * reg

        if type == 'user':
            for u in range(self.user_n):
                latent_matrix[u, :] = solve((FTF + lambdaE), ratings[u, :].dot(fixed_matrix))

        elif type == 'item':
            for i in range(self.item_m):
                latent_matrix[i, :] = solve((FTF + lambdaE), ratings[:, i].T.dot(fixed_matrix))
        return latent_matrix

    '''
    自定义训练迭代次数，正则化因子
    '''

    def train(self, iterations, reg=1):
        times = 0
        while times < iterations:
            # 修正user的隐藏因子矩阵
            self.user_k_n = self.als_function(self.user_k_n, self.item_k_m, self.preference_matrix, reg, type='user')
            # 修正item的隐藏因子矩阵
            self.item_k_m = self.als_function(self.item_k_m, self.user_k_n, self.preference_matrix, reg, type='item')
            times += 1

    '''
    预测用户评分矩阵 n*m
    '''

    def predict(self):
        predict = self.user_k_n.dot(self.item_k_m.T)
        return predict

    '''
    预测用户n对商品m的评分
    '''

    def predict_u_m(self, u, m):
        return self.user_k_n[u, :].dot(self.item_k_m[m, :].T)

    '''
    读取test_index.csv文件，进行预测，返回预测结果
    '''

    def predict_with_index(self, path, iter_times):
        test_index = util.read_file(path, False)
        predict = []
        print("PMF based on ALS train start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        self.train(iter_times)
        print("PMF based on ALS train end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        print("PMF based on ALS predict start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        res = []
        res_matrix = self.predict()
        res.extend(res_matrix)
        for i in range(len(test_index)):
            to_list = []
            to_list.extend(test_index[i])
            u = test_index[i][0]
            m = test_index[i][1]
            m = self.item.index(m)
            to_list.append(res[u][m])
            # to_list.append(self.predict_u_m(u, m))
            predict.append(to_list)
        print("PMF based on ALS predict end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        return predict

    '''
    使用 PMF_ALS model 进行预测
    将预测结果写入csv文件
    '''

    def predict_to_csv(self, test_path, iter_array):

        for i in range(len(iter_array)):
            iter_times = iter_array[i]
            save_path = "out_3.csv"
            # for self_test
            # save_path = "data\\PMF_ALS\\PMF_ALS_predict_kFactors_" + str(self.k_factors) + '_iter_' + str(
            #     iter_times) + ".csv"
            predict = self.predict_with_index(test_path, iter_times)
            util.save_csv_from_rating(predict, save_path)
            print()

    '''
    计算训练集与测试结果的rmse
    iter_array 是一个迭代次数的数组，每一位表示一个迭代次数
    '''

    def calculate_rmse(self, iter_array):
        iter_array.sort()
        iter_times_left = 0
        rmse = []
        for i in range(len(iter_array)):
            iter_times = iter_array[i]
            print('When k_factors is ' + str(self.k_factors) + ', and iterator time is {}'.format(iter_times))
            self.train(iter_times - iter_times_left)
            predict = self.predict()
            train = util.get_rmse(predict, self.preference_matrix)
            rmse.append(train)
            print('RMSE: ' + str(train))
            iter_times_left = iter_times
        return rmse


pmf_als = PMF_ALS()
# '''
iter_times_array = [25]
pmf_als.setup("data\\train.csv", k_factors=500)
pmf_als.predict_to_csv("data\\test_index.csv", iter_times_array)
# '''

'''
for self_test
rmses = []
iter_times_array = [1, 2, 5, 10, 25, 50, 100]
pmf_als.setup("data\\train.csv", k_factors=40)
rmse_40 = pmf_als.calculate_rmse(iter_times_array)
rmses.append([40, rmse_40])

pmf_als.setup("data\\train.csv", k_factors=100)
rmse_100 = pmf_als.calculate_rmse(iter_times_array)
rmses.append([100, rmse_100])

pmf_als.setup("data\\train.csv", k_factors=200)
rmse_200 = pmf_als.calculate_rmse(iter_times_array)
rmses.append([200, rmse_200])

pmf_als.setup("data\\train.csv", k_factors=500)
rmse_500 = pmf_als.calculate_rmse(iter_times_array)
rmses.append([500, rmse_500])

util.draw(iter_times_array, rmses)
'''
