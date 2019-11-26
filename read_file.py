import pandas as pd
train_data = pd.read_csv("E:\\智能系统原理与开发\\Lab2 推荐系统\\RecommendationSystem\\train.csv")
# 构建 Preference Matrix
preference_matrix = pd.pivot_table(train_data, index='userID', columns='itemID',fill_value=0)
print(preference_matrix)
preference_matrix = preference_matrix.values
print(preference_matrix[0][1])
print(len(preference_matrix[0]))
# result : [2967 rows x 7634 columns]
