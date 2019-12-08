import numpy as np
import pandas
from sklearn.metrics import pairwise_distances

import util

preference_matrix, user, item = util.read_file("data\\train.csv")


def computeAdjCosSim(M):
    sim_matrix = np.zeros((M.shape[1], M.shape[1]))
    M_u = M.mean(axis=1)  # means

    for i in range(M.shape[1]):
        for j in range(M.shape[1]):
            if i == j:

                sim_matrix[i][j] = 1
            else:
                if i < j:

                    sum_num = sum_den1 = sum_den2 = 0
                    for k, row in M.loc[:, [i, j]].iterrows():

                        if (M.loc[k, i] != 0) & (M.loc[k, j] != 0):
                            num = (M[i][k] - M_u[k]) * (M[j][k] - M_u[k])
                            den1 = (M[i][k] - M_u[k]) ** 2
                            den2 = (M[j][k] - M_u[k]) ** 2

                            sum_num = sum_num + num
                            sum_den1 = sum_den1 + den1
                            sum_den2 = sum_den2 + den2

                        else:
                            continue

                    den = (sum_den1 ** 0.5) * (sum_den2 ** 0.5)
                    if den != 0:
                        sim_matrix[i][j] = sum_num / den
                    else:
                        sim_matrix[i][j] = 0

                else:
                    sim_matrix[i][j] = sim_matrix[j][i]

    return pandas.DataFrame(sim_matrix)


# print(computeAdjCosSim(pandas.DataFrame(preference_matrix)))


# This function finds k similar items given the item_id and ratings
# matrix M


def k_similar_items(item_id, k):
    sim_matrix = pandas.DataFrame(np.corrcoef(preference_matrix.T))
    m = item.index(item_id)
    similarities = sim_matrix[m]
    similarities.sort_values(ascending=False)
    similarities = similarities[: k].values
    indices = sim_matrix[m].sort_values(ascending=False)[:k].index
    indices = np.array(indices)
    for i in range(k):
        tmp = item[indices[i]]
        indices[i] = tmp
    return similarities, indices


# print(findksimilaritems_adjcos(1, 10))
# indices 保存的是商品的id，不是对应的索引

def predict_u_m(u, m, preference_matrix):
    similarities, indices = k_similar_items(m, 10)
    sum_wt = np.sum(similarities)

    wtd_sum = 0
    for i in range(len(indices)):
        if indices[i] == m:
            continue
        else:
            wtd_sum += preference_matrix.iloc[u, item.index(indices[i])] * (similarities[i])
    prediction = (round(wtd_sum / sum_wt, 1))
    if prediction < 0:
        prediction = 1
    elif prediction > 5:
        prediction = 5
    return prediction


def predict_with_index(path):
    test_index = util.read_file(path, False)
    predict = []
    for i in range(len(test_index)):
        to_list = []
        to_list.extend(test_index[i])
        u = test_index[i][0]
        m = test_index[i][1]
        to_list.append(predict_u_m(u, m, pandas.DataFrame(preference_matrix)))
        predict.append(to_list)

    return predict


def predict_to_csv(test_path, k):
    if k != 0:
        save_path = "data\\item_based\\item_based_predict_k_" + str(k) + ".csv"
        util.save_csv_from_rating(predict_with_index(test_path), save_path)
    else:
        save_path = "data\\item_based\\user_based_predict_without_k.csv"
        # util.save_csv_from_rating(predict_without_k(test_path), save_path)


predict_to_csv("data\\test_index.csv", 30)
