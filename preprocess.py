import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer

# reading files
features_train  = pd.read_csv('./data/dengue_features_train.csv')
labels_train  = pd.read_csv('./data/dengue_labels_train.csv')
features_test  = pd.read_csv('./data/dengue_features_test.csv')

# writing files
# prepro_train_sj = open('./data/Pre-Processed Data/features_train_sj_preprocessed.csv','w')
# prepro_train_iq = open('./data/Pre-Processed Data/features_train_iq_preprocessed.csv','w')
# prepro_test_sj = open('./data/Pre-Processed Data/features_test_sj_preprocessed.csv','w')
# prepro_test_iq = open('./data/Pre-Processed Data/features_test_iq_preprocessed.csv','w')
prepro_train = open('./data/Pre-Processed Data/features_train_preprocessed.csv','w')
prepro_test = open('./data/Pre-Processed Data/features_test_preprocessed.csv','w')

train = pd.merge(labels_train, features_train, on=['city','year','weekofyear'])

# train dataset clustered by city
train_sj = train[train.city == 'sj'].copy()
train_iq = train[train.city == 'iq'].copy()

# test dataset clustered by city
test_sj = features_test[features_test.city == 'sj'].copy()
test_iq = features_test[features_test.city == 'iq'].copy()

# convert to pandas DataFrame objects
train_sj_df = pd.DataFrame(train_sj)
# train_sj_df.file = prepro_train_sj

train_iq_df = pd.DataFrame(train_iq)
# train_iq_df.file = prepro_train_iq

test_sj_df = pd.DataFrame(test_sj)
# test_sj_df.file = prepro_test_sj

test_iq_df = pd.DataFrame(test_iq)
# test_iq_df.file = prepro_test_iq

# array of dataframes
datasets = [train_iq_df, train_sj_df, test_sj_df, test_iq_df]

# train_sj.to_csv(prepro_sj)
# train_iq.to_csv(prepro_iq)

# initialize the imputer
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

# impute values in dataset
# for dataset in datasets:
#     for column in dataset.columns[5:]:
#         feature = pd.DataFrame(dataset[column])
#         print column, pd.DataFrame.mean(feature)
#         try:
#             feature = imp.fit_transform(feature)
#         except(ValueError):
#             pass

results=[]

# simple fillna operation to fill the missing values
for i in range(len(datasets)):
    datasets[i] = datasets[i].fillna(datasets[i].mean())

    if i%2==1:
        results.append(pd.concat(datasets[i-1:i+1], axis=0))

# writing back to files
results[0].to_csv(prepro_train)
results[1].to_csv(prepro_test)
