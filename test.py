import user_based as ub
import item_based as ib
import util
import numpy

target_matrix = util.read_file("test.csv")
print('\n', "Test User Based Recommendation System", '\n')
UB = ub.userBasedRecommSys()
# UB.setup("train_mini_test.csv", 3)
UB.setup("train.csv", 100)
print("preference matrix:")
print(UB.preference_matrix, '\n')
print("user similarity matrix:")
print(UB.user_similarity_matrix, '\n')
print("user average rate array:")
print(UB.average_rate_array, '\n')
print("user k nearest neighbors:")
print(UB.k_nearest_neighbor, '\n')
UB.predicate_rating()
print("predicate rating based on user")
print(UB.preference_matrix, '\n')
print("RMSE:")
print(util.RMSE(numpy.array(UB.preference_matrix), numpy.array(target_matrix)))


print('\n', "Test Item Based Recommendation System", '\n')
IB = ib.itemBasedRecommSys()
IB.setup("train.csv")
print("item similarity matrix:")
print(IB.item_similarity_matrix, '\n')
IB.predicate_rating()
print("predicate rating based on item")
print(IB.preference_matrix)
print("RMSE:")
print(util.RMSE(numpy.array(IB.preference_matrix), numpy.array(target_matrix)))
