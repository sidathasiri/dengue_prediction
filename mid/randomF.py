import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn import preprocessing


train_features = pd.read_csv('./TrainingDataNoClass.csv')
# print(train_features.head())
train_labels = pd.read_csv('./TrainingData.csv')['Class']
# print(train_labels.head())
test_data = pd.read_csv('./TestDataWithHeaders.csv')
# print(test_data.head())
#
#
# #drop unnecessary columns
# # train_features = train_features.drop(columns=['week_start_date'])
# # train_labels = train_labels.drop(columns=['year', 'weekofyear'])
# # test_data = test_data.drop(columns=['week_start_date'])
#
# #mark nan locations
# # train_features = train_features.fillna(-9999)
# # test_data = test_data.fillna(-9999)
#
le = preprocessing.LabelEncoder()
le.fit(train_labels)
# print(train_features.head())
train_labels = pd.DataFrame(le.transform(train_labels))
print(train_labels.head())
# test_data['city'] = le.transform(test_data['city'])
#
#
#split data
X_train, X_test, y_train, y_test = train_test_split(
    train_features, train_labels, test_size=0.2, random_state=0)

clf = RandomForestClassifier(random_state=0,
                             n_estimators=100)
#
#recursive feature elimination
selector = RFE(estimator=clf, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)

# clf = Pipeline([("imputer", Imputer(missing_values=-9999,
#                                           strategy="mean",
#                                           axis=0)),
#                       ("forest", RandomForestClassifier(random_state=0,
#                                                        n_estimators=100))])

clf.fit(X_train, y_train)
#
#
print("Accuracy:", selector.score(X_test, y_test)*100)
result = clf.predict(test_data)
result = le.inverse_transform(result)
result.to_csv('ánswer')
print(result)
# # submission['total_cases'] = selector.predict(test_data)
# # print(submission.head())
# # submission.to_csv('./output/submission.csv', index = False)
# # print("done")
# # print(selector.ranking_)
# # print(train_features.columns.values)
#
# # print(train_features.head(30))