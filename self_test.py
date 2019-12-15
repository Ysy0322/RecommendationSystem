import part1_user_based as ub
import part2_item_based as ib
import datetime
import util


def user_based_predict():
    UB = ub.UserBased()
    print("Test User Based Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    # 考虑k值的基于用户的预测
    UB.setup("data\\train.csv", 3)
    # print("训练集数据所有用户的平均评分：" + str(UB.average_rate_array.mean()))
    UB.predict_for_test("data\\train.csv")
    UB.setup("data\\train.csv", 5)
    UB.predict_for_test("data\\train.csv", False)
    UB.predict_for_test("data\\train.csv")
    UB.setup("data\\train.csv", 10)
    UB.predict_for_test("data\\train.csv")
    UB.setup("data\\train.csv", 20)
    UB.predict_for_test("data\\train.csv")
    UB.setup("data\\train.csv", 50)
    UB.predict_for_test("data\\train.csv")
    UB.setup("data\\train.csv", 100)
    UB.predict_for_test("data\\train.csv")

    # 不考虑k值的基于用户的预测
    # UB.predict_to_csv("data\\test_index.csv", False)

    print("Test User Based Recommendation System end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))


def item_based_predict():
    IB = ib.ItemBased()
    print("Test Item Based Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    # 不考虑k的基于商品的预测
    IB.setup("data\\train.csv", 500)
    IB.predict_for_test("data\\train.csv")
    # 考虑k值的基于商品的预测
    IB.predict_for_test("data\\train.csv", True)
    IB.setup("data\\train.csv", 50)
    IB.predict_for_test("data\\train.csv", True)

    print("Test Item Based Recommendation System end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))


if __name__ == '__main__':
    # user_based_predict()

    # item_based_predict()

    '''
    util.get_rmse_3 = util.get_rmse("data\\self_test\\user_based_test_predict_k_3.csv", "data\\train.csv")
    util.get_rmse_5 = util.get_rmse("data\\self_test\\user_based_test_predict_k_5.csv", "data\\train.csv")
    util.get_rmse_10 = util.get_rmse("data\\self_test\\user_based_test_predict_k_10.csv", "data\\train.csv")
    util.get_rmse_20 = util.get_rmse("data\\self_test\\user_based_test_predict_k_20.csv", "data\\train.csv")
    util.get_rmse_50 = util.get_rmse("data\\self_test\\user_based_test_predict_k_50.csv", "data\\train.csv")
    util.get_rmse_100 = util.get_rmse("data\\self_test\\user_based_test_predict_k_100.csv", "data\\train.csv")
    util.get_rmse_all = util.get_rmse("data\\self_test\\user_based_test_predict_without_k.csv", "data\\train.csv")

    print("user based util.get_rmse when k=3 is: " + str(util.get_rmse_3))
    print("user based util.get_rmse when k=5 is: " + str(util.get_rmse_5))
    print("user based util.get_rmse when k=10 is: " + str(util.get_rmse_10))
    print("user based util.get_rmse when k=20 is: " + str(util.get_rmse_20))
    print("user based util.get_rmse when k=50 is: " + str(util.get_rmse_50))
    print("user based util.get_rmse when k=100 is: " + str(util.get_rmse_100))
    print("user based util.get_rmse when k=all is: " + str(util.get_rmse_all))

    item_util.get_rmse_50 = util.get_rmse("data\\self_test\\item_based_test_predict_k_50.csv", "data\\train.csv")
    item_util.get_rmse_500 = util.get_rmse("data\\self_test\\item_based_test_predict_k_500.csv", "data\\train.csv")
    item_util.get_rmse_all = util.get_rmse("data\\self_test\\item_based_test_predict_cos_without_k.csv", "data\\train.csv")

    print("item based util.get_rmse when k=50 is: " + str(item_util.get_rmse_50))
    print("item based util.get_rmse when k=500 is: " + str(item_util.get_rmse_500))
    print("item based util.get_rmse when k=all is: " + str(item_util.get_rmse_all))
    '''

    '''
                    # for m in range(self.item_m):
                #     if ratings[u][m] != 0.0:
                #         FTF = FTF + fixed_matrix[m].T.dot(fixed_matrix[m])
                #         RI = RI + ratings[u][m] * fixed_matrix[m]
                # latent_matrix[u, :] = np.power(lambdaE+FTF, -1) * (fixed_matrix[m])
                #     # pinv(FTF + lambdaE).dot(RI.T).T
                
                    # for u in range(self.user_n):
                #     if ratings[u][m] != 0.0:
                #         FTF = FTF + fixed_matrix[u].T.dot(fixed_matrix[u])
                #         RI = RI + ratings[u][m] * fixed_matrix[u]
                # latent_matrix[m, :] = pinv(FTF + lambdaE).dot(RI.T).T
    '''
