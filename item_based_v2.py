import datetime

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
        self.user_n = 0
        self.item_m = 0
        self.item_similarity_matrix = []

    def setup(self, path):
        self.preference_matrix_T = util.read_file(path).T
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

    def predicate_rating(self):
        global ab_sim_m12, sim_multi_m12
        for u in range(self.user_n):
            for i in range(self.item_m):
                if self.preference_matrix_T[i][u] == 0.0 or self.preference_matrix_T[i][u] is None:
                    sim_multi_m12 = 0.0
                    ab_sim_m12 = 0.0
                    for m in range(self.item_m):
                        sim_multi_m12 += self.item_similarity_matrix[i][m] * self.preference_matrix_T[m][u]
                        ab_sim_m12 += self.item_similarity_matrix[i][m]
                    if ab_sim_m12 != 0.0:
                        self.preference_matrix_T[i][u] = sim_multi_m12 / ab_sim_m12
        return self.preference_matrix_T


# '''
# Calculate RMSE
# '''
#
# def calculate_rmse(self, path):
#     target_matrix = util.read_file(path)
#     return util.RMSE(numpy.array(self.preference_matrix), numpy.array(target_matrix))

ib = itemBasedRecommSys()
ib.setup("train.csv")
# print(ib.preference_matrix_T)
#
# print(ib.get_similarity_matrix())
# print(ib.item_m)
# print(ib.user_n)
# print(ib.item_similarity_matrix)
print(ib.item_similarity_matrix)
print(len(ib.item_similarity_matrix))
print(len(ib.preference_matrix_T))
print(len(ib.preference_matrix_T[0]))




R = numpy.array([[3, 0.0, 4, 0.0, 0.0],
                 [4.5, 0.0, 0.0, 3.5, 4.0],
                 [0.0, 4.0, 4.0, 4.0, 3.5]])

print(type(R))
similarity = numpy.corrcoef(R)
print(similarity)
mean_user_rating = R.mean(axis=1)  # axis=1 计算每行
print(type(mean_user_rating))


def predict(R, similarity, type='item'):
    prediction = []
    if type == 'user':
        print(mean_user_rating)

        rating_d = numpy.where(R > 0, R - mean_user_rating[:, numpy.newaxis], 0)
        print(rating_d)
        prediction = mean_user_rating[:, numpy.newaxis] + similarity.dot(rating_d) / numpy.array(
            [numpy.abs(similarity).sum(axis=1)]).T
    # elif type == 'item':
    # prediction = rating_d.dot(similarity) / numpy.array([numpy.abs(similarity).sum(axis=1)])
    return prediction


print(predict(numpy.array(R), similarity, 'user'))