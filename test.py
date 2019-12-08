import user_based_v2 as ub
import item_based_v2 as ib
import datetime


def user_based_predicate():
    UB = ub.userBasedRecommSys()
    print("Test User Based Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    UB.setup("data\\train.csv", 5)
    UB.predict_to_csv("data\\test_index.csv")
    # UB.predict_to_csv("data\\test_index.csv", False)

    print("Test User Based Recommendation System end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))


def item_based_predicate():
    IB = ib.itemBasedRecommSys()
    print("Test Item Based Recommendation System start: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    IB.setup("data\\train.csv", 50)
    IB.predict_to_csv("data\\test_index.csv")

    print("Test Item Based Recommendation System end: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))


if __name__ == '__main__':
    user_based_predicate()

    item_based_predicate()
