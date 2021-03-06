import datetime
import numpy

import util

'''
基于用户的推荐系统
① 得到preference_matrix
② 计算两个用户之间的similarity
③ k 个最相近的用户
④ 根据预测公式预测用户对其他商品的评分
'''


class UserBased:
    def __init__(self):

        self.preference_matrix = []
        self.user = []
        self.item = []
        self.user_n = 0
        self.item_m = 0
        self.k_nearest = 0
        self.average_rate_array = []
        self.user_similarity_matrix = []
        self.similarity_uv_k = []
        self.similarity_index_k = []

    def setup(self, path, k):
        self.preference_matrix, self.user, self.item = util.read_file(path)
        self.user_n = len(self.user)
        self.item_m = len(self.item)
        self.k_nearest = k
        self.average_rate_array = self.get_average_rating()
        self.user_similarity_matrix = self.get_user_similarity_matrix()
        self.similarity_uv_k, self.similarity_index_k = self.get_k_neighbors_matrix()

    '''
    计算用户的平均评分矩阵
    [用户u的平均评分, , ,]
    '''

    def get_average_rating(self):
        return self.preference_matrix.sum(1) / (self.preference_matrix != 0).sum(1)

    '''
    计算用户评分的相似度矩阵
    user_n * user_n
    
    第一次是使用numpy 矩阵for循环，结果计算时间太长了
    改用numpy.corrcoef计算用户的Pearson 相关性矩阵
    取值区间为[-1,1]。-1表示完全的负相关，+1表示完全的正相关，0表示没有线性相关。
    皮尔森相关系数描述的是线性关系。严格来说，需要数据集是正态分布的，但不必是零均值的。
    皮尔森相关系数的计算是先对向量每一分量减去分量均值，再求余弦相似度。这一操作称为中心化(将特征向量X根据 x¯\bar x¯x 移动)。
    Pearson 相关性：两个变量的协方差除于两个变量的标准差
    '''

    def get_user_similarity_matrix(self):
        user_similarity_matrix = numpy.corrcoef(self.preference_matrix)
        return user_similarity_matrix

    '''
    计算用户u的前k邻居用户
    返回list[用户的评分], List[用户的index]
    '''

    def k_neighbors(self, u):

        similarity_with_u = [(self.user_similarity_matrix[u][other], other) for other in
                             range(self.user_n)
                             if other != u]
        # 对scores列表排序,从高到底
        similarity_with_u.sort()
        similarity_with_u.reverse()
        similarity_uv_k = []
        similarity_index_k = []
        for sim in range(self.k_nearest):
            similarity_index_k.append(similarity_with_u[sim][1])
            similarity_uv_k.append(similarity_with_u[sim][0])
        return similarity_uv_k, similarity_index_k

    '''
    计算前k邻居用户矩阵 
    user_n * k
    '''

    def get_k_neighbors_matrix(self):
        print("计算前k相似的邻居用户 start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        similarity_uv_k = [0.0] * self.user_n
        similarity_index_k = [0.0] * self.user_n
        for u in range(self.user_n):
            similarity_uv_k[u] = [0.0] * self.k_nearest
            similarity_uv_k_res, similarity_index_k_res = self.k_neighbors(u)
            similarity_uv_k[u] = similarity_uv_k_res
            similarity_index_k[u] = similarity_index_k_res
        print("计算前k相似的邻居用户 end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        return similarity_uv_k, similarity_index_k

    '''
    不过滤用户，使用所有用户的similarity进行预测
    并且对原来存在的数据也进行预测
    '''

    def predict_without_k(self, path):
        print("不考虑k预测 start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        rating_d = numpy.where(numpy.array(self.preference_matrix) > 0,
                               numpy.array(self.preference_matrix) -
                               numpy.array(self.average_rate_array)[:, numpy.newaxis], 0)
        prediction = numpy.array(self.average_rate_array)[:, numpy.newaxis] + numpy.array(
            self.user_similarity_matrix).dot(rating_d) / numpy.array(
            [numpy.abs(numpy.array(self.user_similarity_matrix)).sum(axis=1)]).T
        test_index = util.read_file(path, False)
        predict = []
        for i in range(len(test_index)):
            to_list = []
            to_list.extend(test_index[i])
            u = test_index[i][0]
            m = test_index[i][1]
            rating = prediction[u][self.item.index(m)]
            # rating = round(prediction[u][self.item.index(m)], 1)
            if rating > 5.0:
                rating = 5.0
            to_list.append(rating)
            predict.append(to_list)
        print("不考虑k预测 end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        return predict

    '''
    根据K最相似的邻居用户进行预测
    原来有一张用户和item的稀疏矩阵，从里面扣出来了一些就是现在的测试集，剩下的是发给大家的训练集
    根据test_index中的index对对应的index进行预测
    '''

    def predict_with_k(self, path):
        print("k = " + str(self.k_nearest) + ", predict start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        test_index = util.read_file(path, False)
        predict = []
        for i in range(len(test_index)):
            to_list = []
            to_list.extend(test_index[i])
            u = test_index[i][0]
            m = test_index[i][1]
            average_u = self.average_rate_array[u]
            similarity_u_k = numpy.array(self.similarity_uv_k[u])
            similarity_u_index_k = numpy.array(self.similarity_index_k[u])
            abs_sum_sim = numpy.abs(similarity_u_k).sum()
            sum_mul = 0.0
            if abs_sum_sim == 0:
                to_list.append(0.0)
                predict.append(to_list)
                continue
            else:
                for j in range(self.k_nearest):
                    v = similarity_u_index_k[j]
                    if self.preference_matrix[v][self.item.index(m)] != 0.0:
                        sum_mul += similarity_u_k[j] * \
                                   (self.preference_matrix[v][self.item.index(m)] - self.average_rate_array[v])
                rating = average_u + sum_mul / abs_sum_sim
                if rating > 5.0:
                    rating = 5.0
                to_list.append(rating)
                predict.append(to_list)
        print("k = " + str(self.k_nearest) + ", predict end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
        return predict

    def predict_to_csv(self, test_path, k=True):
        if k:
            save_path = "predict\\out_1.csv"
            # for self_test
            # save_path = "data\\user_based\\user_based_predict_k_" + str(self.k_nearest) + ".csv"
            util.save_csv_from_rating(self.predict_with_k(test_path), save_path)
        else:
            save_path = "data\\user_based\\user_based_predict_without_k.csv"
            util.save_csv_from_rating(self.predict_without_k(test_path), save_path)

    '''
    for self_test
    def predict_for_test(self, test_path, k=True):
        if k:
            save_path = "data\\self_test\\user_based_test_predict_k_" + str(self.k_nearest) + ".csv"
            util.save_csv_from_rating(self.predict_with_k(test_path), save_path)
        else:
            save_path = "data\\self_test\\user_based_test_predict_without_k.csv"
            util.save_csv_from_rating(self.predict_without_k(test_path), save_path)    
    '''
