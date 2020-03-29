# IMPORTS
import pandas as pd
import argparse
import os
import re
from prepare_data import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyts.transformation import BOSS
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample
from tsfresh.transformers import FeatureSelector
from tqdm import tqdm

def extract_ts(X_ts_train, X_ts_test, y_train, y_test, method='PCA', num_samples=5, summarization_int=1):
    num_groups = X_ts_train.shape[1] // (24 // summarization_int)

    # separate into individual (train)
    split_size = 24 // summarization_int
    X_hr_train = X_ts_train[:, :split_size]
    X_resp_train = X_ts_train[:, split_size:split_size*2]
    X_sao2_train = X_ts_train[:, split_size*2:split_size*3]
    X_gcs_train = X_ts_train[:, split_size*3:split_size*4]

    # check if we are also getting nursecharting as time series
    if num_groups > 4:
        X_eyes_train = X_ts_train[:, split_size*4:split_size*5]
        X_verbal_train = X_ts_train[:, split_size*5:split_size*6]
        X_temp_train = X_ts_train[:, split_size*6:split_size*7]

    if num_groups > 7:
        X_systolic_train = X_ts_train[:, split_size*7:split_size*8]
        X_diastolic_train = X_ts_train[:, split_size*8:split_size*9]
        X_meanbp_train = X_ts_train[:, split_size*9:split_size*10]

    # separate into individual (test)
    X_hr_test = X_ts_test[:, :split_size]
    X_resp_test = X_ts_test[:, split_size:split_size*2]
    X_sao2_test = X_ts_test[:, split_size*2:split_size*3]
    X_gcs_test = X_ts_test[:, split_size*3:split_size*4]

    if num_groups > 4:
        X_eyes_test = X_ts_test[:, split_size*4:split_size*5]
        X_verbal_test = X_ts_test[:, split_size*5:split_size*6]
        X_temp_test = X_ts_test[:, split_size*6:split_size*7]

    if num_groups > 7:
        X_systolic_test = X_ts_test[:, split_size*7:split_size*8]
        X_diastolic_test = X_ts_test[:, split_size*8:split_size*9]
        X_meanbp_test = X_ts_test[:, split_size*9:split_size*10]

    # create scaler objects
    hr_scaler = StandardScaler()
    resp_scaler = StandardScaler()
    sao2_scaler = StandardScaler()
    gcs_scaler = StandardScaler()

    if num_groups > 4:
        eyes_scaler = StandardScaler()
        verbal_scaler = StandardScaler()
        temp_scaler = StandardScaler()

    if num_groups > 7:
        systolic_scaler = StandardScaler()
        diastolic_scaler = StandardScaler()
        meanbp_scaler = StandardScaler()

    # fit scaler on train data and transform
    X_hr_train = hr_scaler.fit_transform(X_hr_train)
    X_resp_train = resp_scaler.fit_transform(X_resp_train)
    X_sao2_train = sao2_scaler.fit_transform(X_sao2_train)
    X_gcs_train = gcs_scaler.fit_transform(X_gcs_train)

    if num_groups > 4:
        X_eyes_train = eyes_scaler.fit_transform(X_eyes_train)
        X_verbal_train = verbal_scaler.fit_transform(X_verbal_train)
        X_temp_train = temp_scaler.fit_transform(X_temp_train)

    if num_groups > 7:
        X_systolic_train = systolic_scaler.fit_transform(X_systolic_train)
        X_diastolic_train = diastolic_scaler.fit_transform(X_diastolic_train)
        X_meanbp_train = meanbp_scaler.fit_transform(X_meanbp_train)

    # transform test data
    X_hr_test = hr_scaler.transform(X_hr_test)
    X_resp_test = resp_scaler.transform(X_resp_test)
    X_sao2_test = sao2_scaler.transform(X_sao2_test)
    X_gcs_test = gcs_scaler.transform(X_gcs_test)

    if num_groups > 4:
        X_eyes_test = eyes_scaler.fit_transform(X_eyes_test)
        X_verbal_test = verbal_scaler.fit_transform(X_verbal_test)
        X_temp_test = temp_scaler.fit_transform(X_temp_test)

    if num_groups > 7:
        X_systolic_test = systolic_scaler.fit_transform(X_systolic_test)
        X_diastolic_test = diastolic_scaler.fit_transform(X_diastolic_test)
        X_meanbp_test = meanbp_scaler.fit_transform(X_meanbp_test)

    if method == 'PCA':
        # create PCA objects
        hr_pca = PCA(n_components=num_samples)
        resp_pca = PCA(n_components=num_samples)
        sao2_pca = PCA(n_components=num_samples)
        gcs_pca = PCA(n_components=num_samples)

        hr_feat = ['hr_pca'+str(i) for i in range(num_samples)]
        resp_feat = ['resp_pca'+str(i) for i in range(num_samples)]
        sao2_feat = ['sao2_pca'+str(i) for i in range(num_samples)]
        gcs_feat = ['gcs_pca'+str(i) for i in range(num_samples)]

        if num_groups > 4:
            eyes_pca = PCA(num_samples)
            verbal_pca = PCA(num_samples)
            temp_pca = PCA(num_samples)

            eyes_feat = ['eyes_pca'+str(i) for i in range(num_samples)]
            verbal_feat = ['verbal_pca'+str(i) for i in range(num_samples)]
            temp_feat = ['temp_pca'+str(i) for i in range(num_samples)]

        if num_groups > 7:
            systolic_pca = PCA(num_samples)
            diastolic_pca = PCA(num_samples)
            meanbp_pca = PCA(num_samples)

            systolic_feat = ['systolic_pca'+str(i) for i in range(num_samples)]
            diastolic_feat = ['diastolic_pca'+str(i) for i in range(num_samples)]
            meanbp_feat = ['meanbp_pca'+str(i) for i in range(num_samples)]

        # fit pca on train data and transform
        X_hr_train = hr_pca.fit_transform(X_hr_train)
        X_resp_train = resp_pca.fit_transform(X_resp_train)
        X_sao2_train = sao2_pca.fit_transform(X_sao2_train)
        X_gcs_train = gcs_pca.fit_transform(X_gcs_train)

        if num_groups > 4:
            X_eyes_train = eyes_pca.fit_transform(X_eyes_train)
            X_verbal_train = verbal_pca.fit_transform(X_verbal_train)
            X_temp_train = temp_pca.fit_transform(X_temp_train)

        if num_groups > 7:
            X_systolic_train = systolic_pca.fit_transform(X_systolic_train)
            X_diastolic_train = diastolic_pca.fit_transform(X_diastolic_train)
            X_meanbp_train = meanbp_pca.fit_transform(X_meanbp_train)

        # transform test data
        X_hr_test = hr_pca.transform(X_hr_test)
        X_resp_test = resp_pca.transform(X_resp_test)
        X_sao2_test = sao2_pca.transform(X_sao2_test)
        X_gcs_test = gcs_pca.transform(X_gcs_test)
        #print(hr_pca.components_)

        if num_groups > 4:
            X_eyes_test = eyes_pca.fit_transform(X_eyes_test)
            X_verbal_test = verbal_pca.fit_transform(X_verbal_test)
            X_temp_test = temp_pca.fit_transform(X_temp_test)

        if num_groups > 7:
            X_systolic_test = systolic_pca.fit_transform(X_systolic_test)
            X_diastolic_test = diastolic_pca.fit_transform(X_diastolic_test)
            X_meanbp_test = meanbp_pca.fit_transform(X_meanbp_test)


    elif method == 'sample':
        # figure out how many samples we have
        sample_size = int(np.around(X_hr_train.shape[1] / num_samples))

        # if we rounded down sample_size, add 1 to num_samples
        if X_hr_train.shape[1] > num_samples * sample_size:
            num_samples += 1

        # create arrays to hold sampled values
        X_hr_reduced_train = np.zeros([X_hr_train.shape[0], num_samples])
        X_resp_reduced_train = np.zeros([X_resp_train.shape[0], num_samples])
        X_sao2_reduced_train = np.zeros([X_sao2_train.shape[0], num_samples])
        X_gcs_reduced_train = np.zeros([X_gcs_train.shape[0], num_samples])

        X_hr_reduced_test = np.zeros([X_hr_test.shape[0], num_samples])
        X_resp_reduced_test = np.zeros([X_resp_test.shape[0], num_samples])
        X_sao2_reduced_test = np.zeros([X_sao2_test.shape[0], num_samples])
        X_gcs_reduced_test = np.zeros([X_gcs_test.shape[0], num_samples])

        for i in range(num_samples - 1):
            X_hr_reduced_train[:, i] = np.sum(X_hr_train[:, :i * sample_size])
            X_resp_reduced_train[:, i] = np.sum(X_resp_train[:, :i * sample_size])
            X_sao2_reduced_train[:, i] = np.sum(X_sao2_train[:, :i * sample_size])
            X_gcs_reduced_train[:, i] = np.sum(X_gcs_train[:, :i * sample_size])

            X_hr_reduced_test[:, i] = np.sum(X_hr_test[:, :i * sample_size])
            X_resp_reduced_test[:, i] = np.sum(X_resp_test[:, :i * sample_size])
            X_sao2_reduced_test[:, i] = np.sum(X_sao2_test[:, :i * sample_size])
            X_gcs_reduced_test[:, i] = np.sum(X_gcs_test[:, :i * sample_size])

        # sum the rest of the values for 
        X_hr_reduced_train[:, -1] = np.sum(X_hr_train)
        X_resp_reduced_train[:, -1] = np.sum(X_resp_train)
        X_sao2_reduced_train[:, -1] = np.sum(X_sao2_train)
        X_gcs_reduced_train[:, -1] = np.sum(X_gcs_train)

        X_hr_reduced_test[:, -1] = np.sum(X_hr_test)
        X_resp_reduced_test[:, -1] = np.sum(X_resp_test)
        X_sao2_reduced_test[:, -1] = np.sum(X_sao2_test)
        X_gcs_reduced_test[:, -1] = np.sum(X_gcs_test)

        X_hr_train = X_hr_reduced_train
        X_resp_train = X_hr_reduced_train
        X_sao2_train = X_hr_reduced_train
        X_gcs_train = X_hr_reduced_train

        X_hr_test = X_hr_reduced_test
        X_resp_test = X_hr_reduced_test
        X_sao2_test = X_hr_reduced_test
        X_gcs_test = X_hr_reduced_test

    elif method == 'tsfresh':
        # Selecting train features
        train_features = [X_hr_train, X_resp_train, X_sao2_train, X_gcs_train]
        if num_groups > 4:
            train_features.append(X_eyes_train)
            train_features.append(X_verbal_train)
            train_features.append(X_temp_train)

        if num_groups > 7:
            raise NotImplementedError

        X_presel_train = np.hstack(train_features)
        selector = FeatureSelector()
        selector.fit(X_presel_train, y_train)
        X_ts_train = selector.transform(X_presel_train)

        # Creating feature vector
        tsfresh_feat = ['tsfresh'+str(i) for i in range(X_ts_train.shape[1])]

        # Selecting test features
        test_features = [X_hr_test, X_resp_test, X_sao2_test, X_gcs_test]
        if num_groups > 4:
            test_features.append(X_eyes_test)
            test_features.append(X_verbal_test)
            test_features.append(X_temp_test)

        X_presel_test = np.hstack(test_features)
        X_ts_test = selector.transform(X_presel_test)
        return X_ts_train, X_ts_test
        
    else:
        raise NotImplementedError

    # stack data - train
    train_features = [X_hr_train, X_resp_train, X_sao2_train, X_gcs_train]
    ts_features = [hr_feat, resp_feat, sao2_feat, gcs_feat]
    if num_groups > 4:
        train_features.append(X_eyes_train)
        train_features.append(X_verbal_train)
        train_features.append(X_temp_train)

        ts_features.append(eyes_feat)
        ts_features.append(verbal_feat)
        ts_features.append(temp_feat)

    X_ts_train = np.hstack(train_features)
    ts_features = np.hstack(ts_features)

    # stack data - test
    test_features = [X_hr_test, X_resp_test, X_sao2_test, X_gcs_test]
    if num_groups > 4:
        test_features.append(X_eyes_test)
        test_features.append(X_verbal_test)
        test_features.append(X_temp_test)

    X_ts_test = np.hstack(test_features)

    return X_ts_train, X_ts_test, ts_features

def extract_lab(X_lab_train, X_lab_test, method=None):
    # scaler - fit, transform train data
    lab_scaler = StandardScaler()
    X_lab_train = lab_scaler.fit_transform(X_lab_train)

    # transform test data
    X_lab_test = lab_scaler.transform(X_lab_test)

    if method is None:
        return X_lab_train, X_lab_test

    # apply pca if specified
    elif method == 'PCA':
        # get components of train
        X_lab_cts_train = X_lab_train.iloc[:, :X_lab_train.shape[1]//2]
        X_lab_avgs_train = X_lab_train.iloc[:, :X_lab_train.shape[1]//2]

        # get components of test
        X_lab_cts_test = X_lab_test.iloc[:, :X_lab_test.shape[1]//2]
        X_lab_avgs_test = X_lab_test.iloc[:, :X_lab_test.shape[1]//2]

        # PCA - fit, transform train data 
        lab_cts_pca = PCA(n_components = 10)
        lab_avgs_pca = PCA(n_components = 10)
        X_lab_cts_train = lab_cts_pca.fit_transform(X_lab_cts_train)
        X_lab_avgs_train = lab_avgs_pca.fit_transform(X_lab_cts_train)

        # PCA - transform test data 
        X_lab_cts_test = lab_cts_pca.transform(X_lab_cts_train)
        X_lab_avgs_test = lab_avgs_pca.transform(X_lab_avgs_train)

        # restack data
        X_lab_train = np.hstack([X_lab_cts_train, X_lab_avgs_train])
        X_lab_test = np.hstack([X_lab_cts_test, X_lab_avgs_test])

        return X_lab_train, X_lab_test

def extract_nc(X_nc_train, X_nc_test, method=None):
    # scaler - fit, transform train data
    nc_scaler = StandardScaler()
    X_nc_train = nc_scaler.fit_transform(X_nc_train)

    # transform test data
    X_nc_test = nc_scaler.transform(X_nc_test)

    if method is None:
        return X_nc_train, X_nc_test

def extract_aperiodic(X_aperiodic_train, X_aperiodic_test, method=None):
    # scaler - fit, transform train data
    aperiodic_scaler = StandardScaler()
    X_aperiodic_train = aperiodic_scaler.fit_transform(X_aperiodic_train)

    # transform test data
    X_aperiodic_test = aperiodic_scaler.transform(X_aperiodic_test)

    if method is None:
        return X_aperiodic_train, X_aperiodic_test

    # apply pca if specified
    elif method == 'PCA':
        # get components of train
        X_aperiodic_cts_train = X_aperiodic_train.iloc[:, :X_aperiodic_train.shape[1]//2]
        X_aperiodic_avgs_train = X_aperiodic_train.iloc[:, :X_aperiodic_train.shape[1]//2]

        # get components of test
        X_aperiodic_cts_test = X_aperiodic_test.iloc[:, :X_aperiodic_test.shape[1]//2]
        X_aperiodic_avgs_test = X_aperiodic_test.iloc[:, :X_aperiodic_test.shape[1]//2]

        # PCA - fit, transform train data 
        aperiodic_cts_pca = PCA(n_components = 10)
        aperiodic_avgs_pca = PCA(n_components = 10)
        X_aperiodic_cts_train = aperiodic_cts_pca.fit_transform(X_aperiodic_cts_train)
        X_aperiodic_avgs_train = aperiodic_avgs_pca.fit_transform(X_aperiodic_cts_train)

        # PCA - transform test data 
        X_aperiodic_cts_test = aperiodic_cts_pca.transform(X_aperiodic_cts_train)
        X_aperiodic_avgs_test = aperiodic_avgs_pca.transform(X_aperiodic_avgs_train)

        # restack data
        X_aperiodic_train = np.hstack([X_aperiodic_cts_train, X_aperiodic_avgs_train])
        X_aperiodic_test = np.hstack([X_aperiodic_cts_test, X_aperiodic_avgs_test])

        return X_aperiodic_train, X_aperiodic_test

def extract_med(X_med_train, X_med_test, method=None):
    # scaler - fit, transform train data
    med_scaler = StandardScaler()
    X_med_train = med_scaler.fit_transform(X_med_train)

    # transform test data
    X_med_test = med_scaler.transform(X_med_test)

    if method is None:
        return X_med_train, X_med_test

    raise NotImplementedError

def extract_inf(X_inf_train, X_inf_test, method=None):
    # scaler - fit, transform train data
    inf_scaler = StandardScaler()
    X_inf_train = inf_scaler.fit_transform(X_inf_train)

    # transform test data
    X_inf_test = inf_scaler.transform(X_inf_test)

    if method is None:
        return X_inf_train, X_inf_test

    raise NotImplementedError

def extract_dem(X_dem_train, X_dem_test, method=None):
    # scaler - fit, transform train data
    dem_scaler = StandardScaler()
    X_dem_train = dem_scaler.fit_transform(X_dem_train)

    # transform test data
    X_dem_test = dem_scaler.transform(X_dem_test)

    if method is None:
        return X_dem_train, X_dem_test

    raise NotImplementedError

# TODO fix this to work with different labels
def get_labels(y_gcs = None, mort=False, mort_df = None, dis_loc = False):
    if mort_df is None:
            raise ValueError("Must supply mort_df")
    if y_gcs is None:
        raise ValueError("Must supply y_gcs")

    if mort:
        y_gcs = y_gcs.reset_index().merge(mort_df, on = 'patientunitstayid', how = 'left')
        y = y_gcs['Value'].values
        mortal = y_gcs['unitdischargestatus'].values
        y = np.logical_not(mortal == 'Expired').astype(int)
    elif dis_loc:
        y_gcs = y_gcs.reset_index().merge(mort_df, on = 'patientunitstayid', how = 'left')
        y = y_gcs['Value'].values
        disloc = y_gcs['unitdischargelocation'].values.astype(str)
        a, b = np.unique(disloc, return_counts = True)
        print(a)
        print(b)
        bad = np.array(['Death', 'Skilled Nursing Facility', 'Nursing Home'])
        good = np.array(['Home'])
        dislocNotBad = np.logical_not(np.isin(disloc, bad)).astype(int)
        dislocGood = np.isin(disloc, good).astype(int)
        y = dislocNotBad
    else:   
        y_gcs = y_gcs.reset_index().merge(mort_df, on = 'patientunitstayid', how = 'left')
        y = y_gcs['Value'].values
        mortal = y_gcs['unitdischargestatus'].values
        # modify labels (positive (1) is bad outcome (GCS < 6))
        y[np.logical_or(y < 6, mortal == 'Expired')] = 1
        y[y == 6] = 0

    return y

# TODO also fix this to work with different labels
def resample_data(X_train, y_train, mort=False, method='over'):
    if method == 'over':
        positive = np.argwhere(y_train == 1).flatten()
        negative = np.argwhere(y_train == 0).flatten()

        X_positive = np.hstack([X_train[positive], y_train[positive][:, None]])
        X_negative = np.hstack([X_train[negative], y_train[negative][:, None]])

        X_positive = resample(X_positive, replace=True, n_samples=X_negative.shape[0])
        stacked = np.vstack([X_positive, X_negative])
        np.random.shuffle(stacked)

        X_train = stacked[:, :-1]
        y_train = stacked[:, -1]

    elif method == 'under':
        positive = np.argwhere(y_train == 1).flatten()
        negative = np.argwhere(y_train == 0).flatten()

        X_positive = np.hstack([X_train[positive], y_train[positive][:, None]])
        X_negative = np.hstack([X_train[negative], y_train[negative][:, None]])

        X_negative = resample(X_negative, replace=False, n_samples=X_positive.shape[0])
        stacked = np.vstack([X_positive, X_negative])
        np.random.shuffle(stacked)

        X_train = stacked[:, :-1]
        y_train = stacked[:, -1] 

    else:
        raise ValueError("method must be over or under")

    return X_train, y_train

# TODO implement cross-validation
def train(X, y, model_type='Logistic'):
    if model_type == 'Logistic':
        clf = LogisticRegression(max_iter=1000, penalty='elasticnet', l1_ratio=0.97,
                solver='saga', C=.03)
        params = {'C': [.01, .03, .05, .1],
                'l1_ratio': [.9, .95, .97]}
        
        X = np.where(np.isnan(X), np.ma.array(X, mask=np.isnan(X)).mean(axis=1)[:, np.newaxis], X)
        #clf.fit(X, y)
        gs_log_reg = GridSearchCV(clf, params, cv=3, scoring='accuracy', n_jobs=-1)
        gs_log_reg.fit(X, y)
        #return clf
        return gs_log_reg.best_estimator_

    elif model_type == 'RF':
        clf = RandomForestClassifier(n_estimators = 200, max_depth=6,ccp_alpha = .003, n_jobs=-1)
        params = {'max_depth': [5, 6, 7], 'ccp_alpha': [.001, .003, .005, .007]}
        gs_rf = GridSearchCV(clf, params, cv=5, scoring='accuracy', n_jobs=-1)
        gs_rf.fit(X, y)
        return gs_rf.best_estimator_
        #clf.fit(X,y)
        #return clf
    
    elif model_type == 'NN':
        gcs = False
        if gcs:
            clf = MLPClassifier(max_iter = 10000, solver='lbfgs', alpha=150, hidden_layer_sizes=(5,3))
            params = {'solver': ['lbfgs', 'sgd', 'adam'] }
            gs_nn = GridSearchCV(clf, params, cv = 5, scoring = 'accuracy', n_jobs = -1)
            gs_nn.fit(X, y)
            return gs_nn.best_estimator_
        else:
            clf = MLPClassifier(max_iter = 10000, solver='lbfgs', alpha=40, hidden_layer_sizes=(5,3))
            clf.fit(X,y)
            return clf
    
    raise NotImplementedError

def score(X_train, y_train, X_test, y_test, clf):
    # train metrics
    train_score = clf.score(X_train, y_train)
    y_pred_train = clf.predict_proba(X_train)
    train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_pred_train[:, 1])
    train_auc = roc_auc_score(y_train, y_pred_train[:, 1])

    # test metrics
    test_score = clf.score(X_test, y_test)
    y_pred_test = clf.predict_proba(X_test)
    test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_pred_test[:, 1])
    test_auc = roc_auc_score(y_test, y_pred_test[:, 1])

    return train_score, train_auc, train_thresholds, train_fpr, train_tpr, \
            test_score, test_auc, test_thresholds, test_fpr, test_tpr

def stack_data(rld, reprocess, loaded_loc, processed_loc, data_dir, summarization_int):
    X_ts, X_lab, X_resp_tb, X_med, X_inf, X_dem, X_aperiodic, X_nc, y_discharge, y_mort, y_gcs, patient_list \
            = get_processed_data(loaded_loc, processed_loc, rld, reprocess, data_dir, summarization_int)

    # get number of features (subtract 2 because of patientunitstayid and offset_bin fields)
    num_features = X_ts.shape[1] - 2

    # get individual ts components and stack values horizontally
    X_hr = X_ts[['patientunitstayid', 'offset_bin', 'hr']].pivot(index='patientunitstayid',
            columns='offset_bin', values='hr').values
    X_resp = X_ts[['patientunitstayid', 'offset_bin', 'resp']].pivot(index='patientunitstayid',
            columns='offset_bin', values='resp').values
    X_sao2 = X_ts[['patientunitstayid', 'offset_bin', 'sao2']].pivot(index='patientunitstayid',
            columns='offset_bin', values='sao2').values
    X_gcs = X_ts[['patientunitstayid', 'offset_bin', 'gcs']].pivot(index='patientunitstayid',
            columns='offset_bin', values='gcs').values

    if num_features > 4:
        X_verbal = X_ts[['patientunitstayid', 'offset_bin', 'verbal']].pivot(index='patientunitstayid',
                columns='offset_bin', values='verbal').values
        X_eyes = X_ts[['patientunitstayid', 'offset_bin', 'eyes']].pivot(index='patientunitstayid',
                columns='offset_bin', values='eyes').values
        X_temp = X_ts[['patientunitstayid', 'offset_bin', 'temp']].pivot(index='patientunitstayid',
                columns='offset_bin', values='temp').values

    if num_features > 7:
        X_systolic = X_ts[['patientunitstayid', 'offset_bin', 'noninvasivesystolic']] \
                .pivot(index='patientunitstayid', columns='offset_bin', values='noninvasivesystolic').values
        X_diastolic = X_ts[['patientunitstayid', 'offset_bin', 'noninvasivediastolic']] \
                .pivot(index='patientunitstayid', columns='offset_bin', values='noninvasivediastolic').values
        X_meanbp = X_ts[['patientunitstayid', 'offset_bin', 'noninvasivemean']].pivot(index='patientunitstayid',
                columns='offset_bin', values='noninvasivemean').values

    # get feature names for nursecharting and aperiodic
    nc_feat = X_nc.columns[1:]
    aperiodic_feat = X_aperiodic.columns[1:]
    # including table name as prefix
    nc_feat = ['nc_'+i for i in nc_feat]
    aperiodic_feat = ['aperiodic_'+i for i in aperiodic_feat]

    # put them back into ts numpy array
    if num_features <= 4:
        X_ts = np.hstack([X_hr, X_resp, X_sao2, X_gcs])
        X_aperiodic = X_aperiodic.iloc[:, 1:].values
        X_nc = X_nc.iloc[:, 1:].values

    elif num_features <= 7:
        X_ts = np.hstack([X_hr, X_resp, X_sao2, X_gcs, X_verbal, X_eyes, X_temp])
        X_aperiodic = X_aperiodic.iloc[:, 1:].values

    else:
        X_ts = np.hstack([X_hr, X_resp, X_sao2, X_gcs, X_systolic, X_diastolic, X_meanbp,
            X_eyes, X_verbal, X_temp])

    # get feature names for lab, med, inf, dem, resp
    lab_feat = X_lab.columns[1:]
    med_feat = X_med.columns[1:]
    inf_feat = X_inf.columns[1:]
    dem_feat = X_dem.columns[1:]
    resp_feat = X_resp_tb.columns[1:]

    # appending label corresponding to the table
    lab_feat = ['lab_'+i for i in lab_feat]
    med_feat = ['med_'+str(i) for i in med_feat]
    inf_feat = ['inf_'+i for i in inf_feat]
    dem_feat = ['dem_'+i for i in dem_feat]
    resp_feat = ['resp_'+i for i in resp_feat]

    # get other stuff, move to numpy
    X_lab = X_lab.iloc[:, 1:].values
    X_med = X_med.iloc[:, 1:].values
    X_inf = X_inf.iloc[:, 1:].values
    X_dem = X_dem.iloc[:, 1:].values
    X_resp_tb = X_resp_tb.iloc[:,1:].values

    # combine all Xs to do train test split
    # save length of each component array for doing PCA every run
    num_components = [X_ts.shape[1], X_lab.shape[1], X_med.shape[1], X_inf.shape[1],
            X_dem.shape[1]]

    if num_features <= 4:
        num_components.append(X_nc.shape[1])
        num_components.append(X_aperiodic.shape[1])
        X_stacked = np.hstack([X_ts, X_lab, X_med, X_inf, X_dem, X_nc, X_aperiodic])

    elif num_features <= 7:
        num_components.append(X_aperiodic.shape[1])
        X_stacked = np.hstack([X_ts, X_lab, X_med, X_inf, X_dem, X_aperiodic])

    return X_stacked, y_gcs, num_components, lab_feat, med_feat, inf_feat, dem_feat, resp_feat, nc_feat, aperiodic_feat

def main():
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reload', help="reload data", default=False,
            action='store_true')
    parser.add_argument('-p', '--reprocess', help="reprocess data", default=False,
            action='store_true')
    parser.add_argument('--data_dir', help='path to data directory', default='data')
    parser.add_argument('-n', '--num_runs', help='number of times to bootstrap', default=20)
    parser.add_argument('-s', '--summarization_int', help='binning interval', default=1)
    parser.add_argument('-d', '--debug', help="debug mode, don't write", default=False,
            action='store_true')
    parser.add_argument('-m', dest='method', help="select ts feature extraction method",
            required=False, default='PCA')
    parser.add_argument('-t', dest='type', help="select classification type 'mort' or 'gcs'",
            required=False, default='gcs')
    parser.add_argument('-c', dest='classifier', help="select classifier",
            required=False, default='Logistic')
    args = parser.parse_args()

    # check if loaded/processed dirs exist, check if any files are in them
    existing = os.path.exists(os.path.join(args.data_dir, 'loaded'))
    processed = os.path.exists(os.path.join(args.data_dir, 'loaded', 'processed'))
    results = os.path.exists('model_metrics')

    if not args.debug:
        if not existing:
            # create dir
            os.makedirs(os.path.join(args.data_dir, 'loaded'))

        elif not processed:
            # create dir
            os.makedirs(os.path.join(args.data_dir, 'loaded', 'processed'))

        if not results:
            print("Now on Experiment: 1")
            exp_num = 'exp1'

        else:
            dir_contents = [f for f in os.listdir('model_metrics') if "exp" in f]
            exp_nums = [int(i) for i in [re.findall(r'\d+', i)[0] for i in dir_contents]]
            exp_num = max(exp_nums) + 1
            print("Now on Experiment:", exp_num)
            exp_num = 'exp' + str(exp_num)

    # if there are no files in the loaded dir, we still have to load data
    if len(os.listdir(os.path.join(args.data_dir, 'loaded'))) == 0:
        print("Setting -r and -p flags, no loaded data exists")
        args.reload = True
        args.reprocess = True

    # if there are no files in the processed dir, we still have to process data
    if len(os.listdir(os.path.join(args.data_dir, 'loaded', 'processed'))) == 0:
        print("Setting -p flag, no processed data exists")
        args.reprocess = True

    loaded_dir = os.path.join(args.data_dir, 'loaded')
    processed_dir = os.path.join(args.data_dir, 'loaded', 'processed')
    results_dir = 'model_metrics'

    X_stacked, y_gcs, num_components, lab_feat, med_feat, inf_feat, dem_feat, resp_feat, nc_feat, aperiodic_feat = stack_data(args.reload,
        args.reprocess, loaded_dir, processed_dir, args.data_dir, args.summarization_int)

    mort_df = pd.read_csv(os.path.join(args.data_dir, 'patient_demographics_data.csv')).drop_duplicates(['patientunitstayid'])
    mort = args.type == 'mort'
    y = get_labels(y_gcs, mort, mort_df, False)
    print(mort_df.values)



    # create lists to hold metrics across runs

    # train metrics
    train_thresholds_all = []
    train_auc_all = []
    train_fpr_all = []
    train_tpr_all = []
    train_scores = []

    # test metrics
    test_thresholds_all = []
    test_auc_all = []
    test_fpr_all = []
    test_tpr_all = []
    test_scores = []

    # model metrics
    coeff_all = []
    num_components2 = np.cumsum(num_components)
    print(num_components2)
    print(X_stacked.shape)
    for run in tqdm(range(int(args.num_runs))):
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X_stacked, y, test_size=0.2)

        # get different components (train)
        X_ts_train = X_train[:, :num_components2[0]]
        X_lab_train = X_train[:, num_components2[0]:num_components2[1]]
        X_med_train = X_train[:, num_components2[1]:num_components2[2]]
        X_inf_train = X_train[:, num_components2[2]:num_components2[3]]
        X_dem_train = X_train[:, num_components2[3]:num_components2[4]]

        if len(num_components) > 5:
            X_nc_train = X_train[:, num_components2[4]:num_components2[5]]
            X_aperiodic_train = X_train[:, num_components2[5]:num_components2[6]]

        elif len(num_components) > 4:
            X_aperiodic_train = X_train[:, num_components2[4]:num_components2[5]]


        # get different components (test)
        X_ts_test = X_test[:, :num_components2[0]]
        X_lab_test = X_test[:, num_components2[0]:num_components2[1]]
        X_med_test = X_test[:, num_components2[1]:num_components2[2]]
        X_inf_test = X_test[:, num_components2[2]:num_components2[3]]
        X_dem_test = X_test[:, num_components2[4]:num_components2[4]]

        if len(num_components) > 5:
            X_nc_test = X_test[:, num_components2[4]:num_components2[5]]
            X_aperiodic_test = X_test[:, num_components2[5]:num_components2[6]]

        elif len(num_components) > 4:
            X_aperiodic_test = X_test[num_components2[4]:, :num_components2[5]]

        # extract features
        X_ts_train, X_ts_test, ts_feat = extract_ts(X_ts_train, X_ts_test, y_train, y_test, method=args.method)
        X_lab_train, X_lab_test = extract_lab(X_lab_train, X_lab_test)
        X_med_train, X_med_test = extract_med(X_med_train, X_med_test)
        X_inf_train, X_inf_test = extract_inf(X_inf_train, X_inf_test)

        if len(num_components) > 5:
            X_nc_train, X_nc_test = extract_nc(X_nc_train, X_nc_test)
            X_aperiodic_train, X_aperiodic_test = extract_aperiodic(X_aperiodic_train, X_aperiodic_test)

            # put everything back together
            X_stacked_train = np.hstack([X_ts_train, X_lab_train, X_med_train,
                X_inf_train, X_nc_train, X_aperiodic_train])
            X_stacked_test = np.hstack([X_ts_test, X_lab_test, X_med_test,
                X_inf_test, X_nc_test, X_aperiodic_test])

            # Putting corresponding feature names together
            X_feat = np.hstack([ts_feat, lab_feat, med_feat, inf_feat, nc_feat, aperiodic_feat])

        elif len(num_components) > 4:
            X_aperiodic_train, X_aperiodic_test = extract_aperiodic(X_aperiodic_train, X_aperiodic_test)

            # put everything back together
            X_stacked_train = np.hstack([X_ts_train, X_lab_train, X_med_train,
                X_inf_train, X_aperiodic_train])
            X_stacked_test = np.hstack([X_ts_test, X_lab_test, X_med_test,
                X_inf_test, X_aperiodic_test])

        else:
            # put everything back together
            X_stacked_train = np.hstack([X_ts_train, X_lab_train, X_med_train, X_inf_train])
            X_stacked_test = np.hstack([X_ts_test, X_lab_test, X_med_test, X_inf_test])

        # resample data
        X_stacked_train, y_train = resample_data(X_stacked_train, y_train)

        X_stacked_train = np.where(np.isnan(X_stacked_train), np.ma.array(X_stacked_train, mask=np.isnan(X_stacked_train)).mean(axis=1)[:, np.newaxis], X_stacked_train)
        X_stacked_test = np.where(np.isnan(X_stacked_test), np.ma.array(X_stacked_test, mask=np.isnan(X_stacked_test)).mean(axis=1)[:, np.newaxis], X_stacked_test)
        # train model

        dataLevel = 4
        if dataLevel == 0:
            X_stacked_train = X_stacked_train[:, :15]
            X_stacked_test = X_stacked_test[:, :15]
        elif dataLevel == 1:
            X_stacked_train = X_stacked_train[:, 15:20]
            X_stacked_test = X_stacked_test[:, 15:20]
        elif dataLevel == 2:
            X_stacked_train = X_stacked_train[:, :20]
            X_stacked_test = X_stacked_test[:, :20]
        elif dataLevel == 3:
            X_stacked_train = X_stacked_train[:, :63]
            X_stacked_test = X_stacked_test[:, :63]
        elif dataLevel == 4:
            X_stacked_train = X_stacked_train[:, :]
            X_stacked_test = X_stacked_test[:, :]

        model = train(X_stacked_train, y_train, model_type=args.classifier)

        # get scores
        train_score, train_auc, train_thresholds, train_fpr, train_tpr, \
                test_score, test_auc, test_thresholds, test_fpr, test_tpr = \
                score(X_stacked_train, y_train, X_stacked_test, y_test, model)

        # train metrics
        train_scores.append(train_score)
        train_auc_all.append(train_auc)
        train_fpr_all.append(train_fpr)
        train_tpr_all.append(train_tpr)
        train_thresholds_all.append(train_thresholds)

        # test metrics
        test_scores.append(test_score)
        test_auc_all.append(test_auc)
        test_fpr_all.append(test_fpr)
        test_tpr_all.append(test_tpr)
        test_thresholds_all.append(test_thresholds)

        # model metrics 
        if args.classifier != "RF" and args.classifier != "NN":
            coeff_all.append(model.coef_[:, None].T)

    # stack arrays

    # train metrics
    train_scores=np.vstack(train_scores)
    train_auc_all=np.vstack(train_auc_all)

    # test metrics
    test_scores=np.vstack(test_scores)
    test_auc_all=np.vstack(test_auc_all)

    # model metrics
#    if args.classifier != "RF" and args.classifier != "NN":
#        coeff_all = np.vstack(model.coef_)

    # CHECK THIS
    if args.classifier != 'RF' and args.classifier != 'NN':
        coeff_all = np.asarray(coeff_all)
        coefficients = np.zeros((coeff_all.shape[0], coeff_all.shape[1]))
        coefficients = coeff_all[:,:,0,0]
        coefficients = np.transpose(coefficients)
    if dataLevel == 0:
        X_feat = X_feat[:15]
    elif dataLevel == 1:
        X_feat = X_feat[15:20]
    elif dataLevel == 2:
        X_feat = X_feat[:20]
    elif dataLevel ==3:
        X_feat = X_feat[:63]
    elif dataLevel == 4:
        X_feat = X_feat[:]
    # Creating dataframe to hold feature names and coefficients
    if args.classifier != 'RF' and args.classifier != 'NN':
        coef_df = pd.DataFrame(coefficients, index=list(X_feat))

    print("Train Score:", np.mean(train_scores))
    print("Test Score:", np.mean(test_scores))
    print("Train AUC:", np.mean(train_auc_all))
    print("Test AUC:", np.mean(test_auc_all))

    if not args.debug:
        # make experiment dir
        os.makedirs(os.path.join('model_metrics', exp_num))

        # save coefficients
        if args.classifier != 'RF' and args.classifier != 'NN':
            np.save(os.path.join(results_dir, exp_num, 'coefficients.npy'), coefficients)
            coef_df.to_csv(os.path.join(results_dir, exp_num, 'coefficients.csv'))

        # save files (train)
        np.save(os.path.join(results_dir, exp_num, 'train_scores.npy'), train_scores)
        np.save(os.path.join(results_dir, exp_num, 'train_thresholds_all.npy'), train_thresholds_all)
        np.save(os.path.join(results_dir, exp_num, 'train_auc_all.npy'), train_auc_all)
        np.save(os.path.join(results_dir, exp_num, 'train_fpr_all.npy'), train_fpr_all)
        np.save(os.path.join(results_dir, exp_num, 'train_tpr_all.npy'), train_tpr_all)

        # save files (test)
        np.save(os.path.join(results_dir, exp_num, 'test_scores.npy'), test_scores)
        np.save(os.path.join(results_dir, exp_num, 'test_thresholds_all.npy'), test_thresholds_all)
        np.save(os.path.join(results_dir, exp_num, 'test_auc_all.npy'), test_auc_all)
        np.save(os.path.join(results_dir, exp_num, 'test_fpr_all.npy'), test_fpr_all)
        np.save(os.path.join(results_dir, exp_num, 'test_tpr_all.npy'), test_tpr_all)

if __name__ == "__main__":
    main()
