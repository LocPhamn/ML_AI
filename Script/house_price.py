import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


original_data = pd.read_csv(r"D:\Python plus\House Prices - Advanced Regression Techniques\data\train.csv")
test_data = pd.read_csv(r"D:\Python plus\House Prices - Advanced Regression Techniques\data\test.csv")
submit_sample = pd.read_csv(r"D:\Python plus\House Prices - Advanced Regression Techniques\data\sample_submission.csv")
#handle missing data with fill up all missing data
#in train data
numeric_columns = original_data.select_dtypes(include=['number']).columns
categorical_columns = original_data.select_dtypes(include=['object']).columns

numeric_imputer = SimpleImputer(strategy="median")
original_data[numeric_columns] = numeric_imputer.fit_transform(original_data[numeric_columns])
categorical_imputer = SimpleImputer(strategy="most_frequent")
original_data[categorical_columns] = categorical_imputer.fit_transform(original_data[categorical_columns])

#in test data
test_nums_column = test_data.select_dtypes(include=['number']).columns
test_categorical_column = test_data.select_dtypes(include=['object']).columns
test_data[test_nums_column] = numeric_imputer.fit_transform(test_data[test_nums_column])
test_data[test_categorical_column] = categorical_imputer.fit_transform(test_data[test_categorical_column])

numeric_columns_data = original_data.select_dtypes(include="number")
corr_matrix = numeric_columns_data.corr()
target_corr = corr_matrix['SalePrice']

# Lọc ra các feature có hệ số tương quan lớn hơn một ngưỡng nhất định
threshold = 0.5
high_corr_features = target_corr[target_corr.abs() > threshold].index

# Loại bỏ chính cột 'Target' khỏi danh sách các feature có tương quan cao
high_corr_features = high_corr_features.drop('SalePrice')
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
original_data[["YearBuilt","YearRemodAdd"]]=original_data[["YearBuilt","YearRemodAdd"]].astype("category")
x = original_data[high_corr_features]
y = original_data["SalePrice"]
x_test = test_data[high_corr_features]
y_test = submit_sample.drop("Id",axis= 1)

# x = original_data.drop("SalePrice",axis=1)
# y = original_data["SalePrice"]
# x_test = test_data
# y_test = submit_sample.drop("Id",axis= 1)
preprocess = ColumnTransformer(
transformers=[
    ("num_feature", StandardScaler(), ["OverallQual","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","GarageCars","GarageArea"]),
    ("nom_feature", OneHotEncoder(handle_unknown="ignore"), ["YearBuilt","YearRemodAdd"]),
    # ("num_feature", StandardScaler(), x.select_dtypes(include= "number").columns),
    # ("nom_feature", OneHotEncoder(handle_unknown="ignore"), x.select_dtypes(include= "object").columns),
    # ("num_feature", StandardScaler(), predictor_cols),

]
)
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# x_test = scaler.transform(x_test)
reg = Pipeline(steps=[
    ("preprocessor", preprocess),
    ("model", RandomForestRegressor(n_estimators=800)),
])
reg.fit(x,y)
y_predict = reg.predict(x_test)
print("R2 {}".format(r2_score(y_test,y_predict)))

plt.plot(test_data["Id"].astype("int64"),y_test, color= "blue")
plt.plot(test_data["Id"].astype("int64"),y_predict,color = "red")
plt.show()
# xuất ra file csv
# output = pd.DataFrame({'Id': test_data["Id"].astype("int64"), 'SalePrice': y_predict})
# output.to_csv('submission.csv', index=False)
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.9)
# plt.title('Heatmap of Numerical Data Correlations')
# plt.show()
# num_transformer = Pipeline(steps=[
#     ("imputer", SimpleImputer(missing_values=-1, strategy="median")),
#     ("scaler", StandardScaler())
# ])