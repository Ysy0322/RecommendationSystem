import part1_user_based as ub
import part2_item_based as ib
import part3_PMF_ALS as pmf_als
import part4_optimize as opt
import datetime


def user_based_predict():
    UB = ub.UserBased()
    print("Test User Based Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    UB.setup("data\\train.csv", k=20)
    # 输出路径 predict\\out_1.csv
    UB.predict_to_csv("data\\test_index.csv")
    print("Test User Based Recommendation System end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))


def item_based_predict():
    IB = ib.ItemBased()
    print("Test Item Based Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    IB.setup("data\\train.csv")
    # 输出路径 predict\\out_2.csv
    IB.predict_to_csv("data\\test_index.csv")
    print("Test Item Based Recommendation System end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))


def PMF_ALS_based_predict():
    PMF_ALS = pmf_als.PMF_ALS()
    print("PMF_ALS Based Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    pmf_als = PMF_ALS()

    iter_times_array = [100]
    pmf_als.setup("data\\train.csv", k_factors=10, reg=0.001)
    # 输出路径 predict\\out_3_.csv
    pmf_als.predict_to_csv("data\\test_index.csv", iter_times_array)
    print("PMF_ALS Based Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))


def optimized_predict():
    OPT = opt.ItemBasedOpt()
    print("Optimized Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    OPT.setup("data\\train.csv", 1)
    # 输出路径 predict\\out_4.csv
    OPT.predict_to_csv("data\\test_index.csv")
    print("Optimized Recommendation System end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))


if __name__ == '__main__':
    user_based_predict()

    item_based_predict()

    PMF_ALS_based_predict()

    optimized_predict()
