import pandas as pd
from sklearn.model_selection import train_test_split
from externalLibraries.RR_Enesemble_3rd_party.rr_forest import RRForestClassifier
from sklearn import preprocessing

train_features = pd.read_csv('../data/dengue_features_train.csv')
train_labels = pd.read_csv('../data/dengue_labels_train.csv')
test_data = pd.read_csv('../data/dengue_features_test.csv')

train_features = train_features.drop(columns=['week_start_date'])
train_labels = train_labels.drop(columns=['weekofyear'])
test_data = test_data.drop(columns=['week_start_date'])

train_features = train_features.fillna(-9999)
test_data = test_data.fillna(-9999)

le = preprocessing.LabelEncoder()
le.fit(train_features['city'])
train_features['city'] = le.transform(train_features['city'])
test_data['city'] = le.transform(test_data['city'])

X_train, X_test, y_train, y_test = train_test_split(
    train_features, train_labels, test_size=0.2, random_state=0)


clf = RRForestClassifier(n_estimators=100)
y_train_new = y_train['total_cases']
clf.fit(X_train,y_train_new)

# estimator = Pipeline([("imputer", Imputer(missing_values=-9999,
#                                           strategy="mean",
#                                           axis=0)),
#                       ("forest", RandomForestClassifier(random_state=0,
#                                                        n_estimators=100))])
#
# estimator.fit(X_train, y_train['total_cases'])

print(clf.score(X_test, y_test['total_cases']))

# print(train_features.head(30))