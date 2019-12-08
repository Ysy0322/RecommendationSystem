import datetime

from sklearn.metrics import pairwise_distances

import util
import numpy

'''
基于商品的推荐系统
① 得到preference_matrix
② 计算两个商品之间的similarity
③ 根据预估公式估计用户对其他商品的评分
'''


class itemBasedRecommSys:
    def __init__(self):
        self.preference_matrix_T = []
        self.user = []
        self.item = []
        self.user_n = 0
        self.item_m = 0
        self.item_similarity_matrix = []
        self.k_nearest = 0
        self.similarity_m12_k = []
        self.similarity_index_k = []

    def setup(self, path, k):
        preference_matrix, self.user, self.item = util.read_file(path)
        self.preference_matrix_T = preference_matrix.T
        self.item_m = len(self.preference_matrix_T)
        self.user_n = len(self.preference_matrix_T[0])
        self.k_nearest = k
        self.item_similarity_matrix = self.get_item_similarity_matrix()
        self.similarity_m12_k, self.similarity_index_k = self.get_k_neighbors_matrix()

    '''
    计算商品的相似度矩阵
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
        print("计算前k相似的商品 start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        similarity_m12_k = [0.0] * self.item_m
        similarity_index_k = [0.0] * self.item_m
        for m in range(self.item_m):
            similarity_m12_k[m] = [0.0] * self.k_nearest
            similarity_m12_k_res, similarity_index_k_res = self.k_neighbors(m)
            similarity_m12_k[m] = similarity_m12_k_res
            similarity_index_k[m] = similarity_index_k_res
        print("计算前k相似的商品 end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        return similarity_m12_k, similarity_index_k

    '''
    不考虑k最大相似度的商品预测用户的评分
    '''

    def predict_without_k(self, path):
        print("预测 start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        # for m in range(self.item_m):
        # similarity_m12 = self.item_similarity_matrix[m]
        # if
        prediction = numpy.where(numpy.array(self.preference_matrix_T) > 0,
                                 numpy.array(self.preference_matrix_T).T.dot(self.item_similarity_matrix) / numpy.array(
                                     [numpy.abs(self.item_similarity_matrix).sum(axis=1)]),
                                 )
        # prediction = self.item_similarity_matrix.dot(self.preference_matrix_T) / numpy.array(
        #     numpy.abs(self.item_similarity_matrix).sum())
        # print(len(prediction))
        # print(prediction)
        test_index = util.read_file(path, False)
        predict = []
        for i in range(len(test_index)):
            to_list = []
            to_list.extend(test_index[i])
            u = test_index[i][0]
            m = test_index[i][1]
            # to_list.append(prediction[self.item.index(m)][u])
            to_list.append(prediction[u][self.item.index(m)])
            predict.append(to_list)
        print("预测 end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        return predict

    # 带有k的用户对商品的评分预测

    def predict_with_k(self, path):
        print("预测 start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        test_index = util.read_file(path, False)
        predict = []
        for i in range(len(test_index)):
            to_list = []
            to_list.extend(test_index[i])
            u = test_index[i][0]
            m = test_index[i][1]
            to_list.append(self.predicate_u_m(u, m))
            predict.append(to_list)
        print("预测 end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        print(predict)
        return predict

    # 预测用户u，对商品m的评分

    def predicate_u_m(self, u, m):
        global ab_sim_m12, sim_multi_m12
        pre = 0.0
        sim_multi_m12 = 0.0
        ab_sim_m12 = 0.0
        m = self.item.index(m)
        similarity_m = self.similarity_index_k[m]
        for m1 in range(self.k_nearest):
            m1 = similarity_m[m1]
            if self.preference_matrix_T[m1][u] != 0.0:
                sim_multi_m12 += self.item_similarity_matrix[m][m1] * self.preference_matrix_T[m1][u]
                ab_sim_m12 += numpy.abs(self.item_similarity_matrix[m][m1])
        if ab_sim_m12 != 0.0:
            pre = sim_multi_m12 / ab_sim_m12
        return pre

    # 将预测结果写入csv文件

    def predict_to_csv(self, test_path, k=True):
        if k:
            save_path = "data\\item_based\\item_based_predict_k_" + str(self.k_nearest) + ".csv"
            util.save_csv_from_rating(self.predict_with_k(test_path), save_path)
        else:
            save_path = "data\\item_based\\item_based_predict_without_k.csv"
            util.save_csv_from_rating(self.predict_without_k(test_path), save_path)


IB = itemBasedRecommSys()
print("Test Item Based Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
IB.setup("data\\train.csv", 50)
IB.predict_to_csv("data\\test_index.csv")
