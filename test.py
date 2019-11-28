import user_based as ub
# import item_based as IB

UB = ub.userBasedRecommSys()
UB.setup("train_mini_test.csv", 3)
print("pm:")
# print(UB.preference_matrix)
print("similarity matrix:")
print(UB.get_similarity_matrix())
print("average rate array:")
print(UB.get_average_rate_array())
print("k nearest neighbors:")
print(UB.get_k_nearest_neighbors(3))
print(UB.predicate_rating())
