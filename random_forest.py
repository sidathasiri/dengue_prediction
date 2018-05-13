import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn import preprocessing

train_features = pd.read_csv('./data/dengue_features_train.csv')
train_labels = pd.read_csv('./data/dengue_labels_train.csv')
test_data = pd.read_csv('./data/dengue_features_test.csv')

#calcuate month from week start date
train_features['weekofyear'] = train_features['week_start_date'].apply(lambda date: int(str(date).split('-')[1]))
test_data['weekofyear'] = test_data['week_start_date'].apply(lambda date: int(str(date).split('-')[1]))

#renme the weekofyear to month
train_features = train_features.rename(columns={'weekofyear': 'month'})
test_data = test_data.rename(columns={'weekofyear': 'month'})

#drop unnecessary columns
train_features = train_features.drop(columns=['city', 'week_start_date'])
train_labels = train_labels.drop(columns=['city', 'weekofyear'])
test_data = test_data.drop(columns=['city', 'week_start_date'])

#mark nan locations
train_features = train_features.fillna(-9999)
test_data = test_data.fillna(-9999)


# le = preprocessing.LabelEncoder()
# le.fit(train_features['city'])
# train_features['city'] = le.transform(train_features['city'])
# test_data['city'] = le.transform(test_data['city'])

#split data
X_train, X_test, y_train, y_test = train_test_split(
    train_features, train_labels, test_size=0.2, random_state=0)

clf = RandomForestClassifier(random_state=0,
                             n_estimators=100)

#recursive feature elimination
selector = RFE(estimator=clf, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train['total_cases'])

# clf = Pipeline([("imputer", Imputer(missing_values=-9999,
#                                           strategy="mean",
#                                           axis=0)),
#                       ("forest", RandomForestClassifier(random_state=0,
#                                                        n_estimators=100))])

# clf.fit(X_train, y_train['total_cases'])

print("Accuracy:", selector.score(X_test, y_test['total_cases']))

# print(train_features.head(30))