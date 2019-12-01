import heapq

import util
import numpy

'''
基于用户的推荐系统
① 得到preference_matrix
② 计算两个用户之间的similarity
③ k 个最相近的用户
④ 根据预估公式估计用户对其他商品的评分
'''


class userBasedRecommSys:
    def __init__(self):

        self.preference_matrix = []
        self.user_n = 0
        self.item_m = 0
        self.k_nearest = 0
        self.average_rate_array = []
        self.user_similarity_matrix = []
        self.k_nearest_neighbor = []

    def setup(self, path, k):
        self.preference_matrix = util.read_file(path)
        # rf.read_file("E:\\智能系统原理与开发\\Lab2 推荐系统\\RecommendationSystem\\train.csv")
        self.user_n = len(self.preference_matrix)
        self.item_m = len(self.preference_matrix[0])
        self.k_nearest = k
        self.average_rate_array = self.get_average_rate_array()
        self.user_similarity_matrix = self.get_similarity_matrix()
        self.k_nearest_neighbor = self.get_k_nearest_neighbors(k)

    '''
    计算用户u对所有商品的平均评分
    '''

    def calculate_average_rate(self, u):
        total_rating = 0.0
        rate_num = 0
        for i in range(self.item_m):
            if self.preference_matrix[u][i] != 0.0:
                total_rating += self.preference_matrix[u][i]
                rate_num += 1
        if rate_num == 0:
            return 0.0
        return total_rating / rate_num

    '''
    用户平均评分的数组
    '''

    def get_average_rate_array(self):
        average_rate_array = self.user_n * [0.0]
        for u in range(self.user_n):
            average_rate_array[u] = self.calculate_average_rate(u)
        print(numpy.array(average_rate_array))
        print(len(average_rate_array))
        return average_rate_array

    '''
    计算用户u，v的相似度
    '''

    def calculate_user_similarity(self, u, v):
        diff_sum_u_v = 0.0
        sqrt_sum_u = 0.0
        sqrt_sum_v = 0.0
        average_rate_u = self.calculate_average_rate(u)
        average_rate_v = self.calculate_average_rate(v)
        for i in range(self.item_m):
            if self.preference_matrix[u][i] != 0 and self.preference_matrix[v][i] != 0:
                diff_sum_u_v += (self.preference_matrix[u][i] - average_rate_u) * (
                        self.preference_matrix[v][i] - average_rate_v)
                sqrt_sum_u += (self.preference_matrix[u][i] - average_rate_u) ** 2
                sqrt_sum_v += (self.preference_matrix[v][i] - average_rate_v) ** 2
        if numpy.sqrt(sqrt_sum_u) == 0 or numpy.sqrt(sqrt_sum_v) == 0:
            return 0.0
        similarity = diff_sum_u_v / (numpy.sqrt(sqrt_sum_u) * numpy.sqrt(sqrt_sum_v))
        print(similarity)
        return similarity

    '''
    计算用户的相似性矩阵
    user_similarity_matrix[u][v]表示用户u,v 的相似度
    同一个用户的相似度为1
    '''

    def get_similarity_matrix(self):
        user_similarity_matrix = self.user_n * [0.0]
        for u in range(self.user_n):
            user_similarity_matrix[u] = self.user_n * [0.0]
            for v in range(self.user_n):
                if u != v:
                    user_similarity_matrix[u][v] = self.calculate_user_similarity(u, v)
                else:
                    user_similarity_matrix[u][v] = 1.0
        print(numpy.array(user_similarity_matrix))
        return numpy.array(user_similarity_matrix)

    '''
    k_nearest_neighbors[u][k]
    用户u的前k个相似度最高的用户的index
    row，col都是用户的id
    从高到低排序
    第一位是user自己
    '''

    def get_k_nearest_neighbors(self, k):
        k_nearest_neighbor = self.user_n * [0.0]

        for u in range(self.user_n):
            k_nearest_neighbor[u] = k * [0.0]
            array_u = numpy.array(self.user_similarity_matrix[u])
            k_nearest_neighbor[u] = heapq.nlargest(k, range(len(array_u)), array_u.take)
        print(numpy.array(k_nearest_neighbor))
        return numpy.array(k_nearest_neighbor)

    '''
    预测用户的评分
    '''

    def predicate_rating(self):
        for u in range(self.user_n):
            for i in range(self.item_m):
                if self.preference_matrix[u][i] == 0.0 or self.preference_matrix[u][i] is None:
                    sim_u_v_diff = 0.0
                    ab_sim_u_v = 0.0
                    k = 1
                    while k < self.k_nearest:
                        # for k in range(self.k_nearest):
                        v = self.k_nearest_neighbor[u][k]
                        sim_u_v_diff += self.user_similarity_matrix[u][v] * (
                                self.preference_matrix[v][i] - self.average_rate_array[v])
                        ab_sim_u_v += numpy.abs(self.user_similarity_matrix[u][v])
                        k += 1
                    if ab_sim_u_v != 0.0:
                        self.preference_matrix[u][i] = self.average_rate_array[u] + sim_u_v_diff / ab_sim_u_v
                    # else:
                    # print("ab_sim_u_v is 0")
        return self.preference_matrix
