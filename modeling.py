# IMPORTS
import pandas as pd
import argparse
import os
from prepare_data import *
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample
from tqdm import tqdm

def extract_ts(X_ts_train, X_ts_test, pca_objects = None):
    # separate into individual (train)
    split_size = X_ts_train.shape[1]//4
    X_hr_train = X_ts_train[:, :split_size]
    X_resp_train = X_ts_train[:, split_size:split_size*2]
    X_sao2_train = X_ts_train[:, split_size*2:split_size*3]
    X_gcs_train = X_ts_train[:, split_size*3:]

    # separate into individual (test)
    X_hr_test = X_ts_test[:, :split_size]
    X_resp_test = X_ts_test[:, split_size:split_size*2]
    X_sao2_test = X_ts_test[:, split_size*2:split_size*3]
    X_gcs_test = X_ts_test[:, split_size*3:]

    # create PCA objects
    hr_pca = PCA(n_components=5)
    resp_pca = PCA(n_components=5)
    sao2_pca = PCA(n_components=5)
    gcs_pca = PCA(n_components=10)

    # fit, transform, and restack train data
    X_hr_train = hr_pca.fit_transform(X_hr_train)
    X_resp_train = resp_pca.fit_transform(X_resp_train)
    X_sao2_train = sao2_pca.fit_transform(X_sao2_train)
    X_gcs_train = gcs_pca.fit_transform(X_gcs_train)
    X_ts_train = np.hstack([X_hr_train, X_resp_train, X_sao2_train, X_gcs_train])

    # transform and restack test data
    X_hr_test = hr_pca.transform(X_hr_test)
    X_resp_test = resp_pca.transform(X_resp_test)
    X_sao2_test = sao2_pca.transform(X_sao2_test)
    X_gcs_test = gcs_pca.transform(X_gcs_test)
    X_ts_test = np.hstack([X_hr_test, X_resp_test, X_sao2_test, X_gcs_test])

    return X_ts_train, X_ts_test

def extract_lab(X_lab_train, X_lab_test, method=None):
    if method is None:
        return X_lab_train, X_lab_test

    elif method == 'PCA':
        # get components of train
        X_lab_counts_train = X_lab_train.iloc[:, :X_lab_train.shape[1]//2]
        X_lab_avgs_train = X_lab_train.iloc[:, :X_lab_train.shape[1]//2]

        # get components of test
        X_lab_counts_test = X_lab_test.iloc[:, :X_lab_test.shape[1]//2]
        X_lab_avgs_test = X_lab_test.iloc[:, :X_lab_test.shape[1]//2]

        # PCA - fit, transform train data 
        lab_pca = PCA(n_components = 10)
        X_lab_counts_train = pca.fit_transform(X_lab_counts_train.reshape(-1, 1))
        X_lab_avgs_train = pca.fit_transform(X_lab_avgs_train.reshape(-1, 1))

        # PCA - transform train data 
        X_lab_counts_test = pca.transform(X_lab_counts_train.reshape(-1, 1))
        X_lab_avgs_test = pca.transform(X_lab_avgs_train.reshape(-1, 1))

def extract_med(X_med_train, X_med_test, method=None):
    if method is None:
        return X_med_train, X_med_test

    raise NotImplementedError

def extract_inf(X_inf_train, X_inf_test, method=None):
    if method is None:
        return X_inf_train, X_inf_test

    raise NotImplementedError

def get_labels(final_gcs = None, mort=False, mort_df = None):
    if mort:
        if mort_df is None:
            raise ValueError("Must supply mort_df")
        
        raise NotImplementedError
    
    if final_gcs is None:
        raise ValueError("Must supply final_gcs")
    
    y = final_gcs['Value'].values

    # modify labels (positive (1) is bad outcome (GCS < 6))
    y[y < 6] = 1
    y[y == 6] = 0

    return y

def resample_data(X_train, y_train, mort=False, method='over'):
    if method == 'over':
        if mort:
            positive = np.argwhere(y_train == False).flatten()
            negative = np.argwhere(y_train == True).flatten()
        else:
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
        if mort:
            positive = np.argwhere(y_train == False).flatten()
            negative = np.argwhere(y_train == True).flatten()
        else:
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
        
def train(X, y, model_type='Logistic'):
    if model_type == 'Logistic':
        clf = LogisticRegression(max_iter=2000, penalty='elasticnet', l1_ratio=0.5,
                         solver='saga', C=.5)
        clf.fit(X, y)
        return clf

    raise NotImplementedError

def score(X_train, y_train, X_test, y_test, clf):
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    y_pred_train = clf.predict_proba(X_train)
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_train[:, 0])
    best_thresh = thresholds[np.argmax(np.square(tpr) + np.square(1-fpr))]
    auc = roc_auc_score(y_train, y_pred_train[:, 0])

    return train_score, test_score, best_thresh, auc, thresholds

def main():
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reload', help="reload data", default=False,
            action='store_true')
    parser.add_argument('-b', '--rebin', help="rebin data", default=False,
            action='store_true')
    parser.add_argument('-d', '--data_dir', help='path to data directory', default='data')
    parser.add_argument('-n', '--num_runs', help='number of times to bootstrap', default=10)
    args = parser.parse_args()

    # check if processed dir exists
    existing = os.path.exists(os.path.join(args.data_dir, 'processed'))
    binned = os.path.exists(os.path.join(args.data_dir, 'processed', 'binned'))

    if not existing:
        # create dir
        os.makedirs(os.path.join(args.data_dir, 'processed'))
        assert args.reload, "Must pass the -r flag, no processed data exists"

    if not binned:
        # create dir
        os.makedirs(os.path.join(args.data_dir, 'processed', 'binned'))
        assert args.rebin, "Must pass the -b flag, no binned data exists"

    # get the processed data, if it doesn't exist, we will process it first
    processed_loc = os.path.join(args.data_dir, 'processed')
    X_ts, X_lab, X_med, X_inf, final_gcs = get_processed_data(1.0,
            processed_loc, args.reload, args.rebin, args.data_dir)


    # drop unnamed columns
    X_ts = X_ts.loc[:, ~X_ts.columns.str.contains('^Unnamed')]
    X_lab = X_lab.loc[:, ~X_lab.columns.str.contains('^Unnamed')]
    X_med = X_med.loc[:, ~X_med.columns.str.contains('^Unnamed')]
    X_inf = X_inf.loc[:, ~X_inf.columns.str.contains('^Unnamed')]
    final_gcs = final_gcs.loc[:, ~final_gcs.columns.str.contains('^Unnamed')]

    # drop offset columns
    X_ts = X_ts.loc[:, ~X_ts.columns.str.contains('^offset_bin')]
    X_lab = X_lab.loc[:, ~X_lab.columns.str.contains('^offset_bin')]
    X_med = X_med.loc[:, ~X_med.columns.str.contains('^offset_bin')]
    X_inf = X_inf.loc[:, ~X_inf.columns.str.contains('^offset_bin')]
    final_gcs = final_gcs.loc[:, ~final_gcs.columns.str.contains('^offset_bin')]


    # df to np array

    # get individual ts components
    X_hr = X_ts[['hr']].values.reshape(-1, X_lab.shape[0]).T
    X_resp = X_ts[['resp']].values.reshape(-1, X_lab.shape[0]).T
    X_sao2 = X_ts[['sao2']].values.reshape(-1, X_lab.shape[0]).T
    X_gcs = X_ts[['gcs']].values.reshape(-1, X_lab.shape[0]).T

    # put them back into ts
    X_ts = np.hstack([X_hr, X_resp, X_sao2, X_gcs])

    # get other stuff
    X_lab = X_lab.iloc[:, 1:].values
    X_med = X_med.iloc[:, 1:].values
    X_inf = X_inf.iloc[:, 1:].values

    num_components = [X_ts.shape[1], X_lab.shape[1], X_med.shape[1], X_inf.shape[1]]
    X_stacked = np.hstack([X_ts, X_gcs, X_lab, X_med, X_inf])
    y = get_labels(final_gcs)

    # create lists to hold metrics across runs
    thresholds_all = []
    auc_all = []
    theta_all = []
    coeff_all = []
    test_scores = []
    train_scores = []

    for run in tqdm(range(args.num_runs)):
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X_stacked, y, test_size=0.2)

        # get different components (train)
        X_ts_train = X_train[:, :num_components[0]]
        X_lab_train = X_train[:, :num_components[1]]
        X_med_train = X_train[:, :num_components[2]]
        X_inf_train = X_train[:, :num_components[3]]

        # get different components (test)
        X_ts_test = X_test[:, :num_components[0]]
        X_lab_test = X_test[:, :num_components[1]]
        X_med_test = X_test[:, :num_components[2]]
        X_inf_test = X_test[:, :num_components[3]]

        # extract features
        X_ts_train, X_ts_test = extract_ts(X_ts_train, X_ts_test)
        X_lab_train, X_lab_test = extract_lab(X_lab_train, X_lab_test)
        X_med_train, X_med_test = extract_med(X_med_train, X_med_test)
        X_inf_train, X_inf_test = extract_inf(X_inf_train, X_inf_test)

        # put everything back together
        X_stacked_train = np.hstack([X_ts_train, X_lab_train, X_med_train,
            X_inf_train])
        X_stacked_test = np.hstack([X_ts_test, X_lab_test, X_med_test,
            X_inf_test])

        # resample data
        X_stacked_train, y_train = resample_data(X_stacked_train, y_train)

        # train model
        model = train(X_stacked_train, y_train)

        # get scores
        train_score, test_score, best_thresh, auc, thresholds = score(X_stacked_train,
                y_train, X_stacked_test, y_test, model)

        train_scores.append(train_score)
        test_scores.append(test_score)
        theta_all.append(best_thresh)
        auc_all.append(auc)
        thresholds_all.append(thresholds)

        # get coefficients
        coeff_all.append(model.coef_)

    # stack arrays and save
    train_scores=np.vstack(train_scores)
    test_scores=np.vstack(test_scores)
    theta_all=np.vstack(theta_all)
    auc_all=np.vstack(auc_all)
    print(thresholds_all[0].shape)
    thresholds_all=np.vstack(thresholds_all)

    np.save('train_scores.npy', train_scores)
    np.save('test_scores.npy', test_scores)
    np.save('theta_all.npy', theta_all)
    np.save('auc_all.npy', auc_all)
    np.save('thresholds_all.npy', thresholds_all)


if __name__ == "__main__":
    main()
