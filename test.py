import user_based as ub
import item_based as ib
import util
import datetime


# target_matrix = util.read_file("test.csv")

def user_based_predicate():
    print('\n', "Test User Based Recommendation System", '\n')
    UB = ub.userBasedRecommSys()
    time_stamp = datetime.datetime.now()
    print("User based predicate start: " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
    # UB.setup("train_mini.csv", 3)
    UB.setup("train.csv", 10)
    print("preference matrix:")
    print(UB.preference_matrix, '\n')
    print("user similarity matrix:")
    print(UB.calculate_user_similarity(0, 3))
    print(UB.user_similarity_matrix, '\n')
    print("user average rate array:")
    print(UB.average_rate_array, '\n')
    print("user k nearest neighbors:")
    print(UB.k_nearest_neighbor, '\n')
    UB.predicate_rating()
    time_stamp = datetime.datetime.now()
    print("User based predicate finished: " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
    # print("predicate rating based on user:")
    # print(UB.preference_matrix, '\n')
    print("User Based RMSE:")
    # print(util.RMSE(numpy.array(UB.preference_matrix), numpy.array(target_matrix)))
    time_stamp = datetime.datetime.now()
    print("write into csv started: " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
    util.save_csv(UB.preference_matrix, "user_based_predicate.csv")
    time_stamp = datetime.datetime.now()
    print("write into csv finished: " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))


def item_based_predicate():
    print('\n', "Test Item Based Recommendation System", '\n')
    IB = ib.itemBasedRecommSys()
    time_stamp = datetime.datetime.now()
    print("Item based predicate started: " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
    IB.setup("train.csv")
    print("item similarity matrix:")
    print(IB.item_similarity_matrix, '\n')
    # IB.predicate_rating()
    time_stamp = datetime.datetime.now()
    print("Item based predicate finished: " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
    # print("predicate rating based on item:")
    # print(IB.preference_matrix, '\n')
    print("Item Based RMSE:")
    # print(util.RMSE(numpy.array(IB.preference_matrix), numpy.array(target_matrix)))
    time_stamp = datetime.datetime.now()
    print("write into csv started: " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
    util.save_csv(IB.preference_matrix, "item_based_predicate.csv")
    time_stamp = datetime.datetime.now()
    print("write into csv finished: " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))


# def


if __name__ == '__main__':
    user_based_predicate()

    item_based_predicate()
