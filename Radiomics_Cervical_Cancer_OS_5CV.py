import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import pickle

from sklearn.model_selection import KFold
import torch
from tqdm.notebook import trange

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import lifelines

import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import copy
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index

import torch # For building the networks
import torchtuples as tt # Some useful functionsy
from sklearn.model_selection import StratifiedKFold

from pycox.datasets import metabric
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv


def best_cph_growing_features_v2(fea_df, duration, event, test, selected_features, remaining_features, repeat=10,
                                 folds=5):
    score = []
    val_score = []
    test_score = []  # c-index score
    test_avg_score = []
    test_risk_scores = []  # RISK score
    for feature in remaining_features:
        model_features = selected_features + [feature]
        #         print(f'Model features: {model_features}')
        x_pre_col = fea_df[model_features]
        temp_score_train = np.zeros(repeat)
        temp_score_val = np.zeros(repeat)
        temp_score_test = np.zeros(repeat)

        risk_score = np.zeros(len(test))
        for r in range(repeat):
            cv = KFold(n_splits=folds, shuffle=True, random_state=r)
            # labels = event
            # skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=r)
            fold = 0
            for train_index, val_index in cv.split(x_pre_col):
                fea_train = x_pre_col.loc[train_index]
                event_train = event.loc[train_index]
                duration_train = duration.loc[train_index]
                data = pd.concat([fea_train, event_train, duration_train], axis=1)

                fea_val = x_pre_col.loc[val_index]
                event_val = event.loc[val_index]
                duration_val = duration.loc[val_index]
                try:
                    cf = CoxPHFitter()
                    cf.fit(data, duration_col='survival', event_col='event')
                    temp_score_train[r] += cf.concordance_index_ / folds
                    temp_score_val[r] += concordance_index(
                        event_times=duration_val,
                        #                         predicted_scores = -cf.predict_partial_hazard(fea_val),
                        predicted_scores=cf.predict_expectation(fea_val),
                        event_observed=event_val,
                    ) / folds
                    temp_score_test[r] += concordance_index(
                        event_times=test['survival'],
                        #                         predicted_scores = -cf.predict_partial_hazard(test[model_features]),
                        predicted_scores=cf.predict_expectation(test[model_features]),
                        event_observed=test['event'],
                    ) / folds
                    #                     risk_score += cf.predict_log_partial_hazard(test[model_features])/(repeat*folds)
                    risk_score += cf.predict_expectation(test[model_features]) / (repeat * folds)
                    fold += 1
                except:
                    print('no suitable pair')
                    fold += 1
                    continue

        train_mean_score = np.mean(temp_score_train)
        score.append(train_mean_score)
        val_mean_score = np.mean(temp_score_val)
        val_score.append(val_mean_score)
        test_mean_score = np.mean(temp_score_test)
        test_score.append(test_mean_score)

        avg_score = concordance_index(
            event_times=test['survival'],
            #                         predicted_scores = -risk_score,
            predicted_scores=risk_score,
            event_observed=test['event'],
        )
        test_avg_score.append(avg_score)
        test_risk_scores.append(risk_score)
    #         print(f'Training CIndex: {train_mean_score}, Validation CIndex: {val_mean_score}, Testing CIndex: {test_mean_score}')
    max_id = np.argmax(val_score)
    return remaining_features[max_id], score[max_id], val_score[max_id], test_score[max_id], test_avg_score[max_id], test_risk_scores[max_id]


# We also set some seeds to make this reproducable.
# Note that on gpu, there is still some randomness.
np.random.seed(1234)
_ = torch.manual_seed(123)

maxFea = 4
filetitle = 'Rad_PFS_maxFea4_'

# Load training data
endpoint_data =  pd.read_excel("Cervical Endpoint Overall Survival plus Death 2023 dataset Updated with Deep NotCpltExc NeverDisFreeModified.xlsx")
endpoint_data = endpoint_data.sort_values(by = 'PatientID')
endpoint_data = endpoint_data.reset_index(drop=True)
endpoint_data = endpoint_data.drop(columns=['actual survival'])

# ct_features = pd.read_csv("Cervix_Nii_N4_Norm_1mmRsp_bin50_feature.csv")
ct_features = pd.read_excel("Cervix_Nii_N4_Norm_1mmRsp_bin50_feature_NotCpltExc.xlsx")
ct_features = ct_features.sort_values(by = 'PatientID')
ct_features = ct_features.reset_index(drop=True)
ct_features = ct_features.loc[:, ~ct_features.columns.str.contains('^Unnamed')]

patientID = ct_features.loc[:, "PatientID"]
ct_features = ct_features.drop(columns=['PatientID'])
endpoint_data = endpoint_data.drop(columns=['PatientID'])
endpoint_data['survival'][endpoint_data['survival']>1750] = 1750 # data = ct_features


for randstate in range(5, 15):

    folds = 5
    data = ct_features
    cv = KFold(n_splits=folds, shuffle=True, random_state=randstate)
    Train_index = []
    Val_index = []
    split = cv.split(data)
    for train_index, val_index in split:
        Train_index.append(train_index)
        Val_index.append(val_index)

    # folds = 5
    # data = ct_features
    # labels = endpoint_data['event']
    # Train_index = []
    # Val_index = []
    # skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=randstate)
    # for train_index, val_index in skf.split(data, labels):
    #     Train_index.append(train_index)
    #     Val_index.append(val_index)

    for index in range(0, folds):
        X_train = ct_features.loc[Train_index[index]].reset_index(drop=True)
        X_val = ct_features.loc[Val_index[index]].reset_index(drop=True)
        y_train = endpoint_data.loc[Train_index[index]].reset_index(drop=True)
        y_val = endpoint_data.loc[Val_index[index]].reset_index(drop=True)

        patientID_val = patientID.loc[Val_index[index]].reset_index(drop=True)

        cols_standardize = X_train.columns
        standardize = [([col], StandardScaler()) for col in cols_standardize]  # column normalization
        x_mapper = DataFrameMapper(standardize)

        x_train_norm = x_mapper.fit_transform(X_train).astype('float32')
        x_val_norm = x_mapper.transform(X_val).astype('float32')

        X_train_norm = pd.DataFrame(x_train_norm, columns=X_train.columns)  # put normlized data back to feature table
        X_val_norm = pd.DataFrame(x_val_norm, columns=X_val.columns)

        y_train_norm = pd.DataFrame(y_train[['survival', 'event']].values, columns=['survival', 'event'])
        y_val_norm = pd.DataFrame(y_val[['survival', 'event']].values, columns=['survival', 'event'])

        ## HR analysis
        Rec_norm = y_train_norm['event']
        Dur_norm = y_train_norm['survival']

        score_train = np.zeros([len(X_train_norm.columns), 1])
        HR_train = np.zeros([len(X_train_norm.columns), 1])

        for i in range(len(X_train_norm.columns)):  # for every feature
            temp_train = 0.

            feature = X_train_norm.columns[i]
            data = pd.concat([X_train_norm[feature], Dur_norm, Rec_norm], axis=1)

            cf = CoxPHFitter()
            cf.fit(data, duration_col='survival', event_col='event')
            temp_train = cf.concordance_index_
            hr = cf.hazard_ratios_
            score_train[i, 0] = temp_train
            HR_train[i, 0] = hr
            results = cf.summary
            if i == 0:
                results_pd = results
            else:
                results_pd = pd.concat([results_pd, results], axis=0)

        sorted_pd = results_pd.sort_values('p')
        results_filtered = sorted_pd[sorted_pd['p'] < 0.05]
        columnskeep = ['exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']
        results_save = results_filtered[columnskeep]
        HR_fea = results_save.index

        X_train_norm = X_train_norm[HR_fea.values]
        X_val_norm = X_val_norm[HR_fea.values]

        ## predictive power
        repeat = 10
        folds = 5
        score_test = np.zeros([len(X_val_norm.columns), repeat])
        score_val = np.zeros([len(X_val_norm.columns), repeat])
        score_train = np.zeros([len(X_train_norm.columns), repeat])

        for r in range(0, repeat):
            cv = KFold(n_splits=folds, shuffle=True, random_state=r)
            # labels = y_train_norm['event']
            # skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=r)

            for i in range(0, len(X_train_norm.columns)):  # i for each feature
                temp_val = 0.
                temp_train = 0.
                temp_test = 0.

                feature = X_train_norm.columns[i]
                data = pd.concat([X_train_norm[feature], y_train_norm['survival'], y_train_norm['event']], axis=1)

                for train_index, val_index in cv.split(data):
                    try:
                        cf = CoxPHFitter()
                        cf.fit(data.loc[train_index], duration_col='survival', event_col='event')
                        temp_train = cf.concordance_index_

                        temp_val = concordance_index(
                            event_times=data.loc[val_index]['survival'],
                            #                     predicted_scores = -cf.predict_partial_hazard(data.loc[val_index][[feature]]),
                            predicted_scores=cf.predict_expectation(data.loc[val_index][[feature]]),
                            event_observed=data.loc[val_index]['event'],
                        )

                        temp_test = concordance_index(
                            event_times=y_val_norm['survival'],
                            #                     predicted_scores = -cf.predict_partial_hazard(X_val_norm[[feature]]),
                            predicted_scores=cf.predict_expectation(X_val_norm[[feature]]),
                            event_observed=y_val_norm['event'],
                        )

                        score_train[i, r] += temp_train / folds
                        score_val[i, r] += temp_val / folds
                        score_test[i, r] += temp_test / folds
                    except:
                        print('This feature leads to error: ' + feature)

        mean_train = np.mean(score_train, axis=1)
        std_train = np.std(score_train, axis=1)
        mean_val = np.mean(score_val, axis=1)
        std_val = np.std(score_val, axis=1)
        mean = np.mean(score_test, axis=1)
        std = np.std(score_test, axis=1)

        sort_id_train = sorted(range(len(mean_train)), key=lambda k: mean_train[k], reverse=True)
        sort_id_val = sorted(range(len(mean_val)), key=lambda k: mean_val[k], reverse=True)

        predictive_index = np.where((mean_train > 0.5) & (mean_val > 0.5))[0]
        if len(predictive_index) == 0:
            predictive_index = np.where(mean_train > 0.5)[0]

        X_train_norm = X_train_norm[X_train_norm.columns[predictive_index]]
        X_val_norm = X_val_norm[X_val_norm.columns[predictive_index]]

        ## Correlation anaylsis
        cor = X_train_norm.corr(method='spearman')

        # adjust colorbar
        im_ratio = cor.shape[0] / cor.shape[1]

        remove = np.zeros(len(X_train_norm.columns))
        corrlation = cor.to_numpy()
        for i in range(len(X_train_norm.columns)):
            if remove[i] == 1:
                continue
            for j in range(i + 1, len(X_train_norm.columns)):
                if abs(corrlation[i, j]) > 0.8:
                    remove[j] = 1
        print('There are {} remaining feature in predictive features'.format(len(remove) - sum(remove)))

        col = X_train_norm.columns
        for i in range(len(col)):
            if remove[i] == 1:
                X_train_norm = X_train_norm.drop(col[i], axis=1)
                X_val_norm = X_val_norm.drop(col[i], axis=1)

        ## Step forward feature selection
        max_features = min(10, len(X_train_norm.columns))
        selected_features = []
        remaining_features = X_train_norm.columns.tolist()
        print('Current feature number: ', len(remaining_features))

        grow_pre_train_score = []
        grow_pre_val_score = []
        grow_pre_test_score = []
        grow_pre_avg_test_score = []
        grow_pre_test_risk_score = []

        val = pd.concat([X_val_norm, y_val_norm['survival'], y_val_norm['event']], axis=1)
        for i in range(0, max_features):
            print(f'Growing {i + 1}-th feature...\n')
            new_feature, new_score, new_val_score, new_test_score, new_avg_score, test_risk_score = best_cph_growing_features_v2(
                X_train_norm, y_train_norm['survival'], y_train_norm['event'], val, selected_features, remaining_features,
                repeat=10, folds=5)
            grow_pre_train_score.append(new_score)
            grow_pre_val_score.append(new_val_score)
            grow_pre_test_score.append(new_test_score)
            grow_pre_avg_test_score.append(new_avg_score)
            grow_pre_test_risk_score.append(test_risk_score)
            selected_features.append(new_feature)
            remaining_features.remove(new_feature)
            print('Selected new growing feature: ' + new_feature + ', ci index is: {}'.format(new_score) +
                  ', validation ci index is: {}'.format(new_val_score) + ', testing ci index is: {}'.format(
                new_test_score) +
                  ', testing ci index with avg. risk score is: {}'.format(new_avg_score) + '\n')

        maxval_id = np.argmax(grow_pre_val_score[0:maxFea-1])
        val_risk_score = grow_pre_test_risk_score[maxval_id]

        FeaList = selected_features[0:maxval_id + 1]
        FianlFea = pd.DataFrame(FeaList)

        # write selected feature name to file
        with pd.ExcelWriter(filetitle + str(randstate) + '.xlsx', engine='openpyxl', mode='a') as writer:
            FianlFea.to_excel(writer, sheet_name='SelFea_CT' + str(index))

        C_index_val = concordance_index(
            event_times=y_val_norm['survival'],
            #                 predicted_scores = -val_risk_score.values,
            predicted_scores=val_risk_score.values,
            event_observed=y_val_norm['event'],
        )

        duration = y_val_norm[['survival']]
        event = y_val_norm[['event']]
        cum_event = event
        cum_duration = duration

        Score = pd.DataFrame(val_risk_score.values, columns=['Prediction'])
        score_df = pd.concat([patientID_val, Score, cum_duration, cum_event], axis=1)

        with pd.ExcelWriter(filetitle + str(randstate) + '.xlsx', engine='openpyxl', mode='a') as writer:
            score_df.to_excel(writer, sheet_name='CT' + str(index))




