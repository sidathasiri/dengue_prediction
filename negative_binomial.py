import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn import preprocessing

dengue_features_train  = pd.read_csv('./data/dengue_features_train.csv')
dengue_labels_train  = pd.read_csv('./data/dengue_labels_train.csv')
dengue_features_test  = pd.read_csv('./data/dengue_features_test.csv')
submission = pd.read_csv('./output/submission_format.csv')
dengue_train = pd.merge(dengue_labels_train, dengue_features_train, on=['city','year','weekofyear'])

dengue_train_sj = dengue_train[dengue_train.city == 'sj'].copy()
dengue_train_iq = dengue_train[dengue_train.city == 'iq'].copy()

dengue_test_sj = dengue_features_test[dengue_features_test.city == 'sj'].copy()
dengue_test_iq = dengue_features_test[dengue_features_test.city == 'iq'].copy()

dengue_train_sj.fillna(method='ffill', inplace=True)
dengue_train_iq.fillna(method='ffill', inplace=True)

dengue_test_sj.fillna(method='ffill', inplace=True)
dengue_test_iq.fillna(method='ffill', inplace=True)

sj_correlations = dengue_train_sj.corr()
iq_correlations = dengue_train_iq.corr()

# Remove `week_start_date` string.
dengue_train_sj.drop('reanalysis_tdtr_k', axis=1, inplace=True)
dengue_train_iq.drop('reanalysis_tdtr_k', axis=1, inplace=True)

dengue_train_sj.drop('year', axis=1, inplace=True)
dengue_train_iq.drop('year', axis=1, inplace=True)

dengue_train_sj.drop('ndvi_ne', axis=1, inplace=True)
dengue_train_iq.drop('ndvi_ne', axis=1, inplace=True)

dengue_train_sj.drop('reanalysis_max_air_temp_k', axis=1, inplace=True)
dengue_train_iq.drop('reanalysis_max_air_temp_k', axis=1, inplace=True)

dengue_train_sj.drop('ndvi_se', axis=1, inplace=True)
dengue_train_iq.drop('ndvi_se', axis=1, inplace=True)

dengue_train_sj.drop('station_diur_temp_rng_c', axis=1, inplace=True)
dengue_train_iq.drop('station_diur_temp_rng_c', axis=1, inplace=True)

dengue_train_sj.drop('weekofyear', axis=1, inplace=True)
dengue_train_iq.drop('weekofyear', axis=1, inplace=True)

dengue_train_sj.drop('ndvi_nw', axis=1, inplace=True)
dengue_train_iq.drop('ndvi_nw', axis=1, inplace=True)


# Remove `week_start_date` string.# Remov
dengue_test_sj.drop('reanalysis_tdtr_k', axis=1, inplace=True)
dengue_test_iq.drop('reanalysis_tdtr_k', axis=1, inplace=True)

dengue_test_sj.drop('year', axis=1, inplace=True)
dengue_test_iq.drop('year', axis=1, inplace=True)

dengue_test_sj.drop('ndvi_ne', axis=1, inplace=True)
dengue_test_iq.drop('ndvi_ne', axis=1, inplace=True)

dengue_test_sj.drop('reanalysis_max_air_temp_k', axis=1, inplace=True)
dengue_test_iq.drop('reanalysis_max_air_temp_k', axis=1, inplace=True)

dengue_test_sj.drop('ndvi_se', axis=1, inplace=True)
dengue_test_iq.drop('ndvi_se', axis=1, inplace=True)

dengue_test_sj.drop('station_diur_temp_rng_c', axis=1, inplace=True)
dengue_test_iq.drop('station_diur_temp_rng_c', axis=1, inplace=True)

dengue_test_sj.drop('weekofyear', axis=1, inplace=True)
dengue_test_iq.drop('weekofyear', axis=1, inplace=True)

dengue_test_sj.drop('ndvi_nw', axis=1, inplace=True)
dengue_test_iq.drop('ndvi_nw', axis=1, inplace=True)

sj_train_subtrain = dengue_train_sj.head(800)
sj_train_subtest = dengue_train_sj.tail(dengue_train_sj.shape[0] - 800)

iq_train_subtrain = dengue_train_iq.head(400)
iq_train_subtest = dengue_train_iq.tail(dengue_train_iq.shape[0] - 400)

from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf

from sklearn.cross_validation import train_test_split
import statsmodels.api as sm

from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf


def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "reanalysis_min_air_temp_k + " \
                    "station_min_temp_c + " \
                    "station_max_temp_c + " \
                    "station_avg_temp_c"

    grid = 10 ** np.arange(-8, -3, dtype=np.float64)

    best_alpha = []
    best_score = 1000

    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)

    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model


sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest)
iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest)


sj_predictions = sj_best_model.predict(dengue_test_sj).astype(int)
iq_predictions = iq_best_model.predict(dengue_test_iq).astype(int)

submission = pd.read_csv("./output/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("./output/Model_1_Forecasted_Values.csv")


