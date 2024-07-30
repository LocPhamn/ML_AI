import pandas as pd
from lazypredict.Supervised import LazyClassifier
# import matplotlib.pyplot as plt
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
data = pd.read_csv(r'D:\Python plus\diabetes.csv')
# profile = ProfileReport(data, title="Profiling Report",explorative= True)
# profile.to_file("report.html")
# data.hist()
# data.plot(kind='kde',subplots=True,layout = (3,3),sharex=False)
# plt.show()
# print(data.describe())

#split data to train,test data
target = "Outcome"
x = data.drop(target,axis=1)
y = data[target]

x_train, x_test ,y_train , y_test = train_test_split(x,y, train_size=0.8, random_state= 42) #random seed
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
x_train, x_val ,y_train , y_val =  train_test_split(x_train,y_train, test_size=0.25) # bo val lấy trog bộ train or test

# Data Preprocessing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Model training
# model = RandomForestClassifier()
# model.fit(x_train,y_train)

#Find the best Hyper parameters for estimator or find the best estimator
param1 = {
    'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
}
param2 = {
'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']
}
params = {
    "n_estimators": [100, 200, 300],
    "criterion": ["gini", "entropy", "log_loss"]
}
grid_search = GridSearchCV(SVC(),param_grid=param2, cv=4, verbose=2, scoring='recall', n_jobs=5)
grid_search.fit(x_train,y_train)
print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.best_params_)
y_predict = grid_search.predict(x_test)
x_tt = np
# comparision of true and predict value
# for i,j in zip(y_predict,y_test):
#     print("the predict value {}: the true value {}".format(i,j))
print(classification_report(y_test,y_predict))

#SVC
#       precision    recall  f1-score   support
#
#            0       0.78      0.84      0.81        99
#            1       0.66      0.56      0.61        55
#
#     accuracy                           0.74       154
#    macro avg       0.72      0.70      0.71       154
# weighted avg       0.73      0.74      0.74       154

#logictisRegression
#            0       0.81      0.83      0.82        99
#            1       0.68      0.65      0.67        55

#
#     accuracy                           0.77       154
#    macro avg       0.75      0.74      0.74       154
# weighted avg       0.76      0.77      0.77       154

#RandomForest
#            0       0.82      0.84      0.83        99
#            1       0.70      0.67      0.69        55
#
#     accuracy                           0.78       154
#    macro avg       0.76      0.76      0.76       154
# weighted avg       0.78      0.78      0.78       154

# clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
# models,predictions = clf.fit(x_train, x_test, y_train, y_test)