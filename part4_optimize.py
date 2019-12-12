import datetime
import util
import numpy

'''
对基于商品的推荐系统进行优化
优化方式，修改相似度计算函数
用Pearson 相关性得到商品之间的相似度矩阵
① 得到preference_matrix
② 计算两个商品之间的similarity
③ 根据预估公式估计用户对其他商品的评分
'''


class ItemBasedOpt:
    def __init__(self):
        self.preference_matrix_T = []
        self.user = []
        self.item = []
        self.user_n = 0
        self.item_m = 0
        self.k_nearest = 0
        self.item_similarity_matrix = []
        self.average_rate_array = []
        self.similarity_m12_k = []
        self.similarity_index_k = []

    def setup(self, path, k):
        preference_matrix, self.user, self.item = util.read_file(path)
        self.preference_matrix_T = preference_matrix.T
        self.item_m = len(self.preference_matrix_T)
        self.user_n = len(self.preference_matrix_T[0])
        self.k_nearest = k
        self.average_rate_array = self.get_average_rating()
        self.item_similarity_matrix = self.get_item_similarity_matrix()
        self.similarity_m12_k, self.similarity_index_k = self.get_k_neighbors_matrix()

    '''
    计算用户的平均评分矩阵
    [用户u的平均评分, , ,]
    '''

    def get_average_rating(self):
        return self.preference_matrix_T.sum(1) / (self.preference_matrix_T != 0).sum(1)

    '''
    计算商品的相似度矩阵
    item_m * item_m
    '''

    def get_item_similarity_matrix(self):

        item_similarity_matrix = numpy.corrcoef(self.preference_matrix_T)
        return item_similarity_matrix

    '''
    计算用户u的前k相似商品
    返回list[商品的评分], List[商品的index]
    similarity_index_k 放的是商品的index，不是item_id
    '''

    def k_neighbors(self, m):

        similarity_with_m = [(self.item_similarity_matrix[m][other], other) for other in
                             range(self.item_m)
                             if other != m]
        # 对scores列表排序,从高到底
        similarity_with_m.sort()
        similarity_with_m.reverse()
        similarity_m12_k = []
        similarity_index_k = []
        for sim in range(self.k_nearest):
            similarity_index_k.append(similarity_with_m[sim][1])
            similarity_m12_k.append(similarity_with_m[sim][0])
        return similarity_m12_k, similarity_index_k

    '''
    计算前k相似商品的矩阵 
    item_m * k
    '''

    def get_k_neighbors_matrix(self):
        # print("计算前k相似的商品 start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        similarity_m12_k = [0.0] * self.item_m
        similarity_index_k = [0.0] * self.item_m
        for m in range(self.item_m):
            similarity_m12_k[m] = [0.0] * self.k_nearest
            similarity_m12_k_res, similarity_index_k_res = self.k_neighbors(m)
            similarity_m12_k[m] = similarity_m12_k_res
            similarity_index_k[m] = similarity_index_k_res
        # print("计算前k相似的商品 end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        return similarity_m12_k, similarity_index_k

    '''
    读取test_index.csv文件，进行预测，返回预测结果
    '''

    def predict(self, path, k=False):
        # print("predict start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        test_index = util.read_file(path, False)
        predict = []
        for i in range(len(test_index)):
            to_list = []
            to_list.extend(test_index[i])
            u = test_index[i][0]
            m = test_index[i][1]
            if k:
                to_list.append(self.predict_u_m(u, m, True))
            else:
                to_list.append(self.predict_u_m(u, m, False))
            predict.append(to_list)
        print("predict end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        return predict

    # 预测用户u，对商品m的评分

    def predict_u_m(self, u, m, k=False):
        global ab_sim_m12, sim_multi_m12
        pre = 0.0
        sim_multi_m12 = 0.0
        ab_sim_m12 = 0.0
        m = self.item.index(m)
        if k:
            similarity_m = self.similarity_index_k[m]
            k_n = self.k_nearest
        else:
            similarity_m = self.item_similarity_matrix[m]
            k_n = self.item_m
        for m1 in range(k_n):
            if k:
                m1 = similarity_m[m1]
            if self.preference_matrix_T[m1][u] != 0.0:
                sim_multi_m12 += self.item_similarity_matrix[m][m1] * self.preference_matrix_T[m1][u]
                ab_sim_m12 += numpy.abs(self.item_similarity_matrix[m][m1])
        if ab_sim_m12 != 0.0:
            pre = sim_multi_m12 / ab_sim_m12
        if pre <= 0.0:
            pre = self.average_rate_array[m]
        return pre

    # 将预测结果写入csv文件

    def predict_to_csv(self, test_path, k=False):
        if k:
            save_path = "data\\item_based\\item_based_predict_k_" + str(self.k_nearest) + ".csv"
            util.save_csv_from_rating(self.predict(test_path, True), save_path)
        else:
            save_path = "predict\\out_4.csv"
            util.save_csv_from_rating(self.predict(test_path, False), save_path)

    '''
    for self_test
    def predict_for_test(self, test_path, k=False):
        if k:
            save_path = "data\\self_test\\item_based_test_predict_k_" + str(self.k_nearest) + ".csv"
            util.save_csv_from_rating(self.predict(test_path), save_path)
        else:
            save_path = "data\\self_test\\item_based_test_predict_cos_without_k.csv"
            util.save_csv_from_rating(self.predict(test_path), save_path)
        '''

# IB.predict_for_test("data\\train.csv")
