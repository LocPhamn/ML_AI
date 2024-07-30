import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#take data
data = pd.read_csv(r"D:\Python plus\Titanic - Machine Learning from Disaster\titanic\train.csv")
# profile = ProfileReport(data, title="Profiling Report",explorative= True)
# profile.to_file("report.html")
# print(data.count())

target = "Survived"
useless_feature = ["PassengerId","Name","Ticket","Survived"]
x_train = data.drop(useless_feature,axis=1)
# print(x_train.corr())
y_train = data[target]
x_test = pd.read_csv(r"D:\Python plus\Titanic - Machine Learning from Disaster\titanic\test.csv")
y_test = pd.read_csv(r"D:\Python plus\Titanic - Machine Learning from Disaster\titanic\gender_submission.csv")
y_test = y_test.drop("PassengerId",axis=1)
# print(x_train["Sex"].values_counts())
# fig , ax = plt.subplots(2,3, figsize=(3*3.5,2*3.5))
#
# cols = ["Sex","Embarked","Pclass","SibSp","Parch"]
# for row in range(0,2):
#     for column in range(0,3):
#         index = row*3 + column
#         if index < len(cols):
#             ax_i = ax[row,column]
#             sns.countplot(data=data ,x= cols[index] , hue="Survived", palette="Blues",ax=ax_i)
#             ax_i.set_title(f"Figure {index+1} Survival rate vs {cols[index]}")
#             ax_i.legend(title ='', loc = "upper left", labels = ["not survived","survived"])
# plt.tight_layout()
# plt.show()

# data preprocessing
imp_num = SimpleImputer(missing_values=np.nan , strategy="median")
imp_cate = SimpleImputer(missing_values=np.nan , strategy="most_frequent")

# result1 = imp_num.fit_transform(x_train[["Age"]])
# result2 = imp_cate.fit_transform(x_train[["Cabin"]])
# result3 = imp_cate.fit_transform(x_train[["Embarked"]])
num_process = Pipeline(
    steps=[
        ("handle_missing",SimpleImputer(missing_values=np.nan , strategy="median")),
        ("scaler",StandardScaler())
    ]
)
category_process = Pipeline(
    steps=[
        ("handle_missing",SimpleImputer(missing_values=np.nan , strategy="most_frequent")),
        ("scaler",OneHotEncoder(handle_unknown= "ignore" ))
    ]
)
ct = ColumnTransformer(
    transformers=[
        ("age",num_process, ["Age","Fare"]),
        ("category",category_process, ["Cabin","Embarked","Sex","Pclass"]),
        # ("embarked", SimpleImputer(missing_values=np.nan, strategy="most_frequent"), ["Embarked"])
    ]
)
# result = ct.fit_transform(x_train)
# print(pd.DataFrame(result).corr())
piple = Pipeline(
steps= [
    ("preprocess",ct),
    # ("model",RandomForestClassifier(n_estimators=500,random_state=42))
    ("model",SVC())

]
)
result = piple.fit(x_train,y_train)
y_pred = piple.predict(x_test)
x_tt = np.arange(1,len(x_test)+1)
plt.plot(x_tt,y_pred,color = 'blue')
plt.plot(x_tt,y_test,color = 'red')
plt.show()

# xuất ra file csv
# output = pd.DataFrame({'PassengerId': x_test.PassengerId, 'Survived': y_pred})
# output.to_csv('submission.csv', index=False)
print(classification_report(y_test,y_pred))
# data without sib , parch
#               precision    recall  f1-score   support
#
#            0       0.86      0.90      0.88       266
#            1       0.81      0.75      0.78       152
#
#     accuracy                           0.84       418
#    macro avg       0.84      0.82      0.83       418
# weighted avg       0.84      0.84      0.84       418

# data with sib,parch
#            0       0.86      0.89      0.88       266
#            1       0.80      0.75      0.77       152
#
#     accuracy                           0.84       418
#    macro avg       0.83      0.82      0.82       418
# weighted avg       0.84      0.84      0.84       418