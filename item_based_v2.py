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

    def setup(self, path):
        preference_matrix, self.user, self.item = util.read_file(path)
        self.preference_matrix_T = preference_matrix.T
        self.item_m = len(self.preference_matrix_T)
        self.user_n = len(self.preference_matrix_T[0])
        self.item_similarity_matrix = self.get_item_similarity_matrix()

    '''
    计算商品的相似度矩阵
    '''

    def get_item_similarity_matrix(self):

        item_similarity_matrix = numpy.corrcoef(self.preference_matrix_T)
        return item_similarity_matrix

    '''
    预测用户的评分
    '''

    def predict(self, path):
        print("预测 start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        prediction = self.item_similarity_matrix.dot(self.preference_matrix_T) / numpy.array(
            numpy.abs(self.item_similarity_matrix).sum())
        print(prediction)
        # prediction = self.preference_matrix_T.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        test_index = util.read_file(path, False)
        predict = []
        for i in range(len(test_index)):
            to_list = []
            to_list.extend(test_index[i])
            u = test_index[i][0]
            m = test_index[i][1]
            to_list.append(prediction[self.item.index(m)][u])
            predict.append(to_list)
        print("预测 end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        return predict

    def predict_to_csv(self, test_path):
        save_path = "data\\item_based\\item_based_predict_0.csv"
        # util.save_csv_from_rating(self.predict(test_path), save_path)
        util.save_csv_from_rating(self.predict_0(test_path), save_path)

    def predict_0(self, path):
        print("预测 start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        test_index = util.read_file(path, False)
        predict = []
        for i in range(len(test_index)):
            to_list = []
            to_list.extend(test_index[i])
            u = test_index[i][0]
            m = test_index[i][1]
            to_list.append(self.predicate_rating(u, m))
            predict.append(to_list)
        print("预测 end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        print(predict)
        return predict

    def predicate_rating(self, u, m):
        global ab_sim_m12, sim_multi_m12
        pre = 0.0
        sim_multi_m12 = 0.0
        ab_sim_m12 = 0.0
        m = self.item.index(m)
        for m1 in range(self.item_m):
            # m1 = self.item.index(m1)
            if self.preference_matrix_T[m][u] != 0.0:
                sim_multi_m12 += self.item_similarity_matrix[m][m1] * self.preference_matrix_T[m][u]
                ab_sim_m12 += numpy.abs(self.item_similarity_matrix[m][m1])
        if ab_sim_m12 != 0.0:
            pre = sim_multi_m12 / ab_sim_m12
        return pre


ib = itemBasedRecommSys()
ib.setup("data\\train.csv")
ib.predict_to_csv("data\\test_index.csv")
