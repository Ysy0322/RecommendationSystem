import user_based_v2 as ub
import item_based_v2 as ib
import datetime


def user_based_predict():
    UB = ub.userBasedRecommSys()
    print("Test User Based Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    # 考虑k值的基于用户的预测
    UB.setup("data\\train.csv", 5)
    print(UB.item)
    UB.predict_to_csv("data\\test_index.csv")
    # 不考虑k值的基于用户的预测
    UB.predict_to_csv("data\\test_index.csv", False)

    print("Test User Based Recommendation System end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))


def item_based_predict():
    IB = ib.itemBasedRecommSys()
    print("Test Item Based Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    # 不考虑k的基于商品的预测
    IB.setup("data\\train.csv", 500)
    IB.predict_to_csv("data\\test_index.csv")
    # 考虑k值的基于商品的预测
    # IB.predict_to_csv("data\\test_index.csv", True)

    print("Test Item Based Recommendation System end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))


if __name__ == '__main__':

    user_based_predict()

    item_based_predict()
