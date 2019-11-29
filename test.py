import user_based as ub
import item_based as ib
import util
import numpy
import pandas

# import DataFrame

# target_matrix = util.read_file("test.csv")
print('\n', "Test User Based Recommendation System", '\n')
UB = ub.userBasedRecommSys()
UB.setup("train_mini_test.csv", 3)
# UB.setup("train.csv", 100)
print("preference matrix:")
print(UB.preference_matrix, '\n')
print("user similarity matrix:")
print(UB.user_similarity_matrix, '\n')
print("user average rate array:")
print(UB.average_rate_array, '\n')
print("user k nearest neighbors:")
print(UB.k_nearest_neighbor, '\n')
print("predicate rating based on user")
UB.predicate_rating()
print(UB.preference_matrix, '\n')
util.save_csv(UB.preference_matrix, "user_based_predicate.csv")

print("RMSE:")
# print(util.RMSE(numpy.array(UB.preference_matrix), numpy.array(target_matrix)))


print('\n', "Test Item Based Recommendation System", '\n')
IB = ib.itemBasedRecommSys()
IB.setup("train.csv")
print("item similarity matrix:")
print(IB.item_similarity_matrix, '\n')
print("predicate rating based on item")
# IB.predicate_rating()
print(IB.preference_matrix)
util.save_csv(IB.preference_matrix, "item_based_predicate.csv")
print("RMSE:")
# print(util.RMSE(numpy.array(IB.preference_matrix), numpy.array(target_matrix)))
