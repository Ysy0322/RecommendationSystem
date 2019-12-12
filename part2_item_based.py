import datetime
import util
import numpy

'''
用时十几分钟
基于商品的推荐系统
① 得到preference_matrix
② 计算两个商品之间的similarity
③ 根据预估公式估计用户对其他商品的评分
'''


class ItemBased:
    def __init__(self):
        self.preference_matrix_T = []
        self.user = []
        self.item = []
        self.user_n = 0
        self.item_m = 0
        self.item_similarity_matrix = []
        self.average_rate_array = []

    def setup(self, path):
        preference_matrix, self.user, self.item = util.read_file(path)
        self.preference_matrix_T = preference_matrix.T
        self.item_m = len(self.preference_matrix_T)
        self.user_n = len(self.preference_matrix_T[0])
        self.average_rate_array = self.get_average_rating()
        self.item_similarity_matrix = self.get_item_similarity_matrix()

    '''
    计算用户的平均评分矩阵
    [用户u的平均评分, , ,]
    '''

    def get_average_rating(self):
        return self.preference_matrix_T.sum(1) / (self.preference_matrix_T != 0).sum(1)

    '''
    计算商品的相似度矩阵 余弦函数计算
    item_m * item_m
    '''

    def get_item_similarity_matrix(self):
        col = len(self.item)
        item_similarity_matrix = numpy.zeros((col, col))
        for i in range(col):
            for j in range(col):
                if i <= j < col:
                    sim_i_j = numpy.dot(self.preference_matrix_T[i], self.preference_matrix_T[j]) / (
                            numpy.linalg.norm(self.preference_matrix_T[i]) * (
                        numpy.linalg.norm(self.preference_matrix_T[j])))
                    item_similarity_matrix[i][j] = sim_i_j
                    item_similarity_matrix[j][i] = sim_i_j
        return item_similarity_matrix

    '''
    读取test_index.csv文件，进行预测，返回预测结果
    '''

    def predict(self, path):
        print("predict start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        test_index = util.read_file(path, False)
        predict = []
        for i in range(len(test_index)):
            to_list = []
            to_list.extend(test_index[i])
            u = test_index[i][0]
            m = test_index[i][1]
            to_list.append(self.predict_u_m(u, m))
            predict.append(to_list)
        print("predict end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        return predict

    # 预测用户u，对商品m的评分

    def predict_u_m(self, u, m):
        global ab_sim_m12, sim_multi_m12
        pre = 0.0
        sim_multi_m12 = 0.0
        ab_sim_m12 = 0.0
        m = self.item.index(m)
        for m1 in range(self.item_m):
            if self.preference_matrix_T[m1][u] != 0.0:
                sim_multi_m12 += self.item_similarity_matrix[m][m1] * self.preference_matrix_T[m1][u]
                ab_sim_m12 += numpy.abs(self.item_similarity_matrix[m][m1])
        if ab_sim_m12 != 0.0:
            pre = sim_multi_m12 / ab_sim_m12
        if pre <= 0.0:
            pre = self.average_rate_array[m]
        return pre

    # 将预测结果写入csv文件

    def predict_to_csv(self, test_path):
        save_path = "predict\\out_2.csv"
        util.save_csv_from_rating(self.predict(test_path), save_path)

    '''
    for test
    def predict_for_test(self, test_path):
        save_path = "data\\self_test\\item_based_test_predict_cos.csv"
        util.save_csv_from_rating(self.predict(test_path), save_path)
    '''


'''
for test
IB.predict_for_test("data\\train.csv")
print("Item Based RMSE is: " + str(get_rmse_with_csv("data\\self_test\\item_based_test_predict_cos.csv", "data\\train.csv")))
'''
