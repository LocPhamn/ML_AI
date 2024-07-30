import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile
from imblearn.over_sampling import RandomOverSampler,SMOTEN
def regexLocation(document):
    result = re.findall("[A-Z]{2}",document)
    if(len(result)!= 0):
        return result[0]
    else:
        return document

data = pd.read_excel("final_project.ods", engine="odf",dtype = str)
# print(data["career_level"].value_counts())
target = "career_level"
data["location"] = data["location"].apply(regexLocation)

#phân chia dữ liệu
data = data.dropna() # do có đúng 1 dòng nan nên drop
x = data.drop(target,axis=1)
y = data[target]
# print(data.isna().sum())
x_train, x_test, y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
print(y_train.value_counts())
print("===========")
# xử lý dữ liệu & tối ưu dữ liệu

# ros = RandomOverSampler(random_state=42,sampling_strategy={
#     "director_business_unit_leader": 1000 ,
#     "specialist":500,
#     "managing_director_small_medium_company":500
# })

ros = SMOTEN(random_state=42,k_neighbors=2, sampling_strategy={
    "director_business_unit_leader": 1000 ,
    "specialist":500,
    "managing_director_small_medium_company":500
})
x_train, y_train = ros.fit_resample(x_train,y_train)
print(y_train.value_counts())
job_level = ['specialist','senior_specialist_or_project_manager','bereichsleiter','manager_team_leader', 'director_business_unit_leader','managing_director_small_medium_company']
col_trans = ColumnTransformer(
    transformers =[
        ("title",TfidfVectorizer(stop_words="english" , ngram_range=(1,1)),"title"),
        # ValueError: Found unknown categories ['West Virginia', 'South Carolina'] in column 0 during transform
        # dữ liệu trong bộ test có mà train k có
        ("location",OneHotEncoder(handle_unknown="ignore"),["location"]),
        ("description", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),min_df=0.01 ,max_df=0.95), "description"),
        ("function", OneHotEncoder(), ["function"]),
        ("industry", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry"),
    ]
)
#Nháp
# vectorizer = TfidfVectorizer(stop_words="english" , ngram_range=(1,2)) # loại bỏ các stopwords và chia sentences thành các token
# result = vectorizer.fit_transform(x_train["title"])
# result = col_trans.fit_transform(x_train)
# st = result.todense() # xuất ra dạng vecto của result
# print(result.shape)
# print(st)
# print(data["career_level"].unique())
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# print(result.shape)

# Model
cls = Pipeline(
    steps= [
        ("preprocessing", col_trans),
        # ("selectkBest", SelectKBest(chi2,k=500)), # tối ưu tốc độ bằng cách chọn 500sample để train
        ("selectPercentile",SelectPercentile(chi2,percentile=10)),  # tối ưu tốc độ bằng cách chọn 10%sample để train
        ("model",DecisionTreeClassifier())
    ]
)
result = cls.fit(x_train,y_train)
y_pred = cls.predict(x_test)

print(classification_report(y_test,y_pred))


# result without min , max df
#     accuracy                           0.66      1615
#    macro avg       0.42      0.42      0.42      1615
# weighted avg       0.66      0.66      0.66      1615

# result with min , max df
#   accuracy                           0.67      1615
#  macro avg       0.43      0.42      0.42      1615
# weighted avg       0.67      0.67      0.67      1615


# result with max,min df,vơ 500 feature
#   accuracy                           0.68      1615
#  macro avg       0.41      0.39      0.39      1615
# weighted avg       0.67      0.68      0.67      1615