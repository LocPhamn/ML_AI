import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from ydata_profiling import ProfileReport
from lazypredict.Supervised import LazyRegressor


data = pd.read_csv("StudentScore.xls", delimiter=",")
# profile = ProfileReport(data, title="Score Report", explorative=True)
# profile.to_file("score.html")

target = "writing score"

x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(missing_values=-1, strategy="median")),
    ("scaler", StandardScaler())
])

education_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree",
                    "master's degree"]
gender_values = ["male", "female"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False))
])

# fit và transform các cột với từng hàm biến đổi tương ứng.
preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["reading score", "math score"]),
    ("ord_feature", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_feature", nom_transformer, ["race/ethnicity"]),
])

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
    # ("model2",RandomForestRegressor())
])

params = {
    "preprocessor__num_feature__imputer__strategy":["median","mean"],
    "model__n_jobs": [1,2,3,4,5],
}

# clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
# models,predictions = clf.fit(x_train, x_test, y_train, y_test)
x_tt = np.arange(1,201)
reg.fit(x_train,y_train)
y_predict = reg.predict(x_test)
plt.plot(x_tt,y_predict,color = 'blue')
plt.plot(x_tt,y_test,color = 'red')
plt.show()

# grid_search = GridSearchCV(estimator=reg,param_grid=params,cv= 5,scoring="r2",verbose=2)
# grid_search.fit(x_train,y_train)
# print(grid_search.best_estimator_)
# print(grid_search.best_score_)
# print(grid_search.best_params_)

# với dữ liệu quá lớn và tham số gridSearchCV sẽ thực hiện rất lâu
# với tham số n_jobs trong gridSearchCV sẽ chỉnh được số nhân xử lý -> tăng tốc độ xử lý
# grid_search = GridSearchCV(estimator=reg,param_grid=params,cv= 5,scoring="r2",verbose=2, n_jobs= 6)
# grid_search.fit(x_train,y_train)
# print(grid_search.best_estimator_)
# print(grid_search.best_score_)
# print(grid_search.best_params_)

# RandomizedSearchCV là phương pháp như GridSearchCV nhưng thời gian thực hiện ngắn
# ngựợc lại với GridSearchCV. RandomizedSearchCV không thử hết tất cả các tham số , chỉ lấy số lượng tham số theo chỉ đinh n_iters
# và ngẫu nhiên
randomized_search = RandomizedSearchCV(reg,param_distributions=params, cv=5,scoring="r2",verbose= 2 , n_iter=1)
randomized_search.fit(x_train,y_train)
print(randomized_search.best_estimator_)
print(randomized_search.best_score_)
print(randomized_search.best_params_)

reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
for i, j in zip(y_predict, y_test):
    print("Predicted value: {}. Actual value: {}".format(i, j))
print(r2_score(y_test,y_predict))
