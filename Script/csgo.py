import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, make_scorer ,recall_score
from sklearn.model_selection import GridSearchCV


data = pd.read_csv("csgo.csv")
# print(data.describe())
# print(data.head(10))

# profile = ProfileReport(data, title="CSGO Report",explorative= True) # trực quan hóa và statistic data
# profile.to_file("csgoReport.html")
# data["result"]=data["result"].apply(lambda x:0 if x=="Lost" else (1 if x=="Win" else 2)) # chuyen cot result thanh 0 và 1

#split feature
target = "result"
unNecessary_feature = ["map","day","month","year",	"date",	"wait_time_s",	"match_time_s",	"team_a_rounds","team_b_rounds","result"]
x = data.drop(unNecessary_feature,axis=1)
# print(x.corr())
y = data[target]
print(y.count())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Handle data
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

#Model
model = SVC()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

#Model with Best parameters
svc_param = {
    "kernel":["linear","poly","rbf","sigmoid"],
    # "degree" : [1,2,3,4,5],
}
ran_params = {
    "n_estimators": [100, 200, 300],
    "criterion": ["gini", "entropy", "log_loss"]
}
# print(y_test)
grid_search = GridSearchCV(RandomForestClassifier(),param_grid=ran_params,verbose=2,cv=5, scoring='accuracy' ,n_jobs=5)
grid_search.fit(x_train,y_train)
# print(grid_search.best_estimator_)
# print(grid_search.best_score_)
# print(grid_search.best_params_)

#proceed
model = grid_search.best_estimator_
y_pred_grid = model.predict(x_test)
# print("predict value = {} true value {}".format(y_pred_grid,y_test[898]))
print(classification_report(y_test,y_pred_grid))
# svc(kernal = 'linear')
#       precision    recall  f1-score   support
#
#         Lost       0.68      0.84      0.75       105
#          Tie       0.00      0.00      0.00        21
#          Win       0.78      0.75      0.77       101
#
#     accuracy                           0.72       227
#    macro avg       0.49      0.53      0.51       227
# weighted avg       0.66      0.72      0.69       227