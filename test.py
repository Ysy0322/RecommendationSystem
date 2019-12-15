import part1_user_based as ub
import part2_item_based as ib
import part3_PMF_ALS as pmf_als
import part4_optimize as opt
import datetime
import util


def user_based_predict():
    UB = ub.UserBased()
    print("Test User Based Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    UB.setup("data\\train.csv", k=20)
    # 输出路径 predict\\out_1.csv
    UB.predict_to_csv("data\\test_index.csv")
    print("Test User Based Recommendation System end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + '\n')


def item_based_predict():
    IB = ib.ItemBased()
    print("Test Item Based Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    IB.setup("data\\train.csv")
    # 输出路径 predict\\out_2.csv
    IB.predict_to_csv("data\\test_index.csv")
    print("Test Item Based Recommendation System end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + '\n')


def PMF_ALS_based_predict():
    PMF_ALS = pmf_als.PMF_ALS()
    print("PMF_ALS Based Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    iter_times_array = [10]
    PMF_ALS.setup("data\\train.csv", k_factors=800, reg=0.01)
    # 输出路径 predict\\out_3.csv
    PMF_ALS.predict_to_csv("data\\test_index.csv", iter_times_array)
    print("PMF_ALS Based Recommendation System end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + '\n')


def optimized_predict():
    OPT = opt.ItemBasedOpt()
    print("Optimized Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    OPT.setup("data\\train.csv", 1)
    # 输出路径 predict\\out_4.csv
    OPT.predict_to_csv("data\\test_index.csv")
    print("Optimized Recommendation System end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + '\n')


if __name__ == '__main__':
    user_based_predict()

    item_based_predict()

    optimized_predict()

    PMF_ALS_based_predict()

    '''
    for test
    util.filter_res("predict\\out_1.csv", "predict\\out_1(2).csv")
    util.filter_res("predict\\out_2.csv", "predict\\out_2(2).csv")
    util.filter_res("predict\\out_3.csv", "predict\\out_3(2).csv")
    util.filter_res("predict\\out_4.csv", "predict\\out_4(2).csv")
    '''
