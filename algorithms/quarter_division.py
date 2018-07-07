import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from externalLibraries.RR_Enesemble_3rd_party.rr_forest import RRForestClassifier


def get_city_data(data_path,labels_path=None):
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
#     assuming each quater consist of 13 weeks
    df.fillna(method='ffill', inplace=True)
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)

    # seperate the dataset by two cities
    sj = df.loc['sj']
    iq = df.loc['iq']
    return sj,iq

def get_quater_data(df,qtr):
    df['week_start_date'] = pd.to_datetime(df['week_start_date']).dt.date
    df['Qtr'] = pd.to_datetime(df['week_start_date']).dt.quarter
    qtr_data = df.loc[df['Qtr']==qtr]
    return qtr_data

def pre_process_data(df,isTrain=True):



    # df = df[features]
    # df=df.drop(['Qtr','week_start_date'],axis=1)
    #
    # y_data = df['total_cases']
    # x_data = df.drop(['total_cases'], axis=1)
    if isTrain:
        features = ['reanalysis_specific_humidity_g_per_kg',
                    'reanalysis_dew_point_temp_k',
                    'station_avg_temp_c',
                    'station_min_temp_c',
                    'total_cases'
                    ]
        df = df[features]
        # df=df.drop(['Qtr','week_start_date'],axis=1)
        #
        y_data = df['total_cases']
        x_data = df.drop(['total_cases'], axis=1)
        # x_train,x_test,y_train,y_test = train_test_split(x_data,y_data)
        # return x_train,y_train,x_test,y_test
        return x_data,y_data
    else:
        features = ['reanalysis_specific_humidity_g_per_kg',
                    'reanalysis_dew_point_temp_k',
                    'station_avg_temp_c',
                    'station_min_temp_c'
                    ]

        df = df[features]
        return df



def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']
    df = df[features]

    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)

    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']
    y_sj =sj['total_cases']
    y_iq =iq['total_cases']
    x_sj= sj.drop(['total_cases'],axis=1)
    x_iq= iq.drop(['total_cases'],axis=1)
    return y_sj ,y_iq ,x_sj,x_iq

sj_all_train,iq_all_train = get_city_data('../data/dengue_features_train.csv',labels_path='../data/dengue_labels_train.csv')
sj_all_test,iq_all_test= get_city_data('../data/dengue_features_test.csv')

sj_predictions= []
for n in range (1,5):
    sj_train = get_quater_data(sj_all_train, n)
    sj_test = get_quater_data(sj_all_test,n)
    # x_sj_train,y_sj_train,x_sj_test,y_sj_test= pre_process_data(sj_train)
    x_sj_train,y_sj_train= pre_process_data(sj_train,True)
    sj_test_new=pre_process_data(sj_test,False)
    clf=RandomForestClassifier()
    # clf = MLPClassifier(n_neighbors=5)
    clf.fit(x_sj_train,y_sj_train)
    # sj_pred=clf.predict(x_sj_test)
    sj_pred=clf.predict(sj_test_new)
    sj_test_new['total_cases'] = sj_pred.T
    # sj_test_new['city']='sj'
    # total_cases=pd.DataFrame(sj_pred.T,columns='total_cases')
    # sj_test.add(total_cases)
    sj_predictions.append(sj_test_new)
    # sj_train.total_cases.plot(ax=axes[0], label="Actual")
    # print("Accuracy for sj ",n, " Quater" , accuracy_score(y_sj_test,sj_pred))

test_results_sj=pd.concat(sj_predictions)
sorted_test_results_sj= test_results_sj.sort_index()
# sort_test_results=test_results.sort_index()

iq_predictions=[]
for n in range (1,5):
    iq_train = get_quater_data(iq_all_train, n)
    iq_test = get_quater_data(iq_all_test,n)
    # x_iq_train,y_iq_train,x_iq_test,y_iq_test= pre_process_data(iq_train)
    x_iq_train,y_iq_train= pre_process_data(iq_train,True)
    iq_test_new=pre_process_data(iq_test,False)
    clf=RandomForestClassifier(n_estimators=50)
    # clf = MLPClassifier(n_neighbors=5)
    clf.fit(x_iq_train,y_iq_train)
    # iq_pred=clf.predict(x_iq_test)
    iq_pred=clf.predict(iq_test_new)
    iq_test_new['total_cases'] = iq_pred.T
    # iq_test_new['city']='iq'
    # total_cases=pd.DataFrame(iq_pred.T,columns='total_cases')
    # iq_test.add(total_cases)
    iq_predictions.append(iq_test_new)
    # iq_train.total_cases.plot(ax=axes[0], label="Actual")
    # print("Accuracy for sj ",n, " Quater" , accuracy_score(y_sj_test,sj_pred))

test_results_iq=pd.concat(iq_predictions)
sorted_test_results_iq= test_results_iq.sort_index()

final_test_results=pd.concat([sorted_test_results_sj,sorted_test_results_iq])

# indexed_final_test_results=final_test_results.set_index('city',append=True)
# sorted_final_test_results= indexed_final_test_results.sort_index()

submission = pd.read_csv("../output/submission_format.csv",
                         index_col=[0, 1, 2])
submission.total_cases = final_test_results['total_cases'].values.T
submission.to_csv("../output/quater_yd_1.csv")


print("")

# for n in range (1,5):
#     iq_first = get_quater_data(iq_all_train, n)
#     # x_iq_train,y_iq_train,x_iq_test,y_iq_test= pre_process_data(iq_first)
#     x_iq_train,y_iq_train= pre_process_data(iq_first,False)
#     clf=RandomForestClassifier(n_estimators=50)
#     # clf = KNeighborsClassifier(n_neighbors=5)
#     clf.fit(x_iq_train,y_iq_train)
#     iq_pred_first=clf.predict(x_iq_test)
#     # print("Accuracy for iq ",n, " Quater" , accuracy_score(y_iq_test,iq_pred_first))



# fot sj
# X_train_sj, X_test_sj, y_train_sj, y_test_sj = train_test_split(x_sj, y_sj)
# clf = RandomForestClassifier(random_state=0,
#                              n_estimators=20)
# clf = AdaBoostClassifier(n_estimators =50)
# clf = KNeighborsClassifier(n_neighbors=10)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(5,7,10), random_state=1)
# clf = KNeighborsClassifier(n_neighbors=10)
# clf.fit(X_train_sj,y_train_sj)
# sj_pred = clf.predict(X_test_sj)
# print("Accuracy for sj" , accuracy_score(y_test_sj,sj_pred))
# # for iq
# X_train_iq, X_test_iq, y_train_iq, y_test_iq = train_test_split( x_iq, y_iq)
# clf = RandomForestClassifier(random_state=0,
#                              n_estimators=20)
# clf = AdaBoostClassifier(n_estimators =50)
# clf = KNeighborsClassifier(n_neighbors=10)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(5,7,10), random_state=1)
# clf = KNeighborsClassifier(n_neighbors=10)
# clf.fit(X_train_iq,y_train_iq)
# iq_pred = clf.predict(X_test_iq)
# print("Accuracy for iq" , accuracy_score(y_test_iq,iq_pred))

