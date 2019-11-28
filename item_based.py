import util
import numpy

'''
基于商品的推荐系统
① 得到preference_matrix
② 计算两个商品之间的similarity
③ 
④ 根据预估公式估计用户对其他商品的评分
'''


class itemBasedRecommSys:
    def __init__(self):
        self.preference_matrix = []
        self.user_n = 0
        self.item_m = 0
        # self.average_rate_array = []
        self.item_similarity_matrix = []

    def setup(self, path):
        self.preference_matrix = util.read_file(path)
        # rf.read_file("E:\\智能系统原理与开发\\Lab2 推荐系统\\RecommendationSystem\\train.csv")
        self.user_n = len(self.preference_matrix)
        self.item_m = len(self.preference_matrix[0])
        self.item_similarity_matrix = self.get_similarity_matrix()

    def calculate_item_similarity(self, m1, m2):
        diff_u_m1_m2 = 0.0
        u_m1 = 0.0
        u_m2 = 0.0
        for u in range(self.user_n):
            # self.preference_matrix[u][m1] is not None and self.preference_matrix[u][m2] is not None and
            if self.preference_matrix[u][m1] != 0.0 and self.preference_matrix[u][m2] != 0.0:
                diff_u_m1_m2 += self.preference_matrix[u][m1] * self.preference_matrix[u][m2]
                u_m1 += self.preference_matrix[u][m1] ** 2
                u_m2 += self.preference_matrix[u][m2] ** 2
            if numpy.sqrt(u_m1) == 0.0 or numpy.sqrt(u_m2) == 0.0:
                return 0.0
            similarity = diff_u_m1_m2 / (numpy.sqrt(u_m1) * numpy.sqrt(u_m2))
            return similarity

    '''
    计算商品的相似性矩阵
    item_similarity_matrix[m1][m2]表示用户m1,m2 的相似度
    同一个商品的相似度为1
    '''

    def get_similarity_matrix(self):
        item_similarity_matrix = self.item_m * [0.0]
        for m1 in range(self.item_m):
            item_similarity_matrix[m1] = self.item_m * [0.0]
            for m2 in range(self.item_m):
                if m1 != m2:
                    item_similarity_matrix[m1][m2] = self.calculate_item_similarity(m1, m2)
                else:
                    item_similarity_matrix[m1][m2] = 1.0

        return item_similarity_matrix
