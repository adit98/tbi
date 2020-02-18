# IMPORTS
import pandas as pd
import argparse
import os
import re
from prepare_data import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample
from tqdm import tqdm

def extract_ts(X_ts_train, X_ts_test):
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

    # create scaler objects
    hr_scaler = StandardScaler()
    resp_scaler = StandardScaler()
    sao2_scaler = StandardScaler()
    gcs_scaler = StandardScaler()

    # fit scaler on train data and transform
    X_hr_train = hr_scaler.fit_transform(X_hr_train)
    X_resp_train = resp_scaler.fit_transform(X_resp_train)
    X_sao2_train = sao2_scaler.fit_transform(X_sao2_train)
    X_gcs_train = gcs_scaler.fit_transform(X_gcs_train)

    # transform test data
    X_hr_test = hr_scaler.transform(X_hr_test)
    X_resp_test = resp_scaler.transform(X_resp_test)
    X_sao2_test = sao2_scaler.transform(X_sao2_test)
    X_gcs_test = gcs_scaler.transform(X_gcs_test)

    # create PCA objects
    hr_pca = PCA(n_components=5)
    resp_pca = PCA(n_components=5)
    sao2_pca = PCA(n_components=5)
    gcs_pca = PCA(n_components=10)

    # fit pca on train data and transform
    X_hr_train = hr_pca.fit_transform(X_hr_train)
    X_resp_train = resp_pca.fit_transform(X_resp_train)
    X_sao2_train = sao2_pca.fit_transform(X_sao2_train)
    X_gcs_train = gcs_pca.fit_transform(X_gcs_train)

    # stack data
    X_ts_train = np.hstack([X_hr_train, X_resp_train, X_sao2_train, X_gcs_train])

    # transform test data
    X_hr_test = hr_pca.transform(X_hr_test)
    X_resp_test = resp_pca.transform(X_resp_test)
    X_sao2_test = sao2_pca.transform(X_sao2_test)
    X_gcs_test = gcs_pca.transform(X_gcs_test)

    # restack
    X_ts_test = np.hstack([X_hr_test, X_resp_test, X_sao2_test, X_gcs_test])

    return X_ts_train, X_ts_test

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
                         solver='saga', C=.8)
        clf.fit(X, y)
        return clf

    raise NotImplementedError

def score(X_train, y_train, X_test, y_test, clf):
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    y_pred_train = clf.predict_proba(X_train)
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_train[:, 0])
    best_thresh = thresholds[np.argmax(np.square(tpr) + np.square(1-fpr))]
    auc = roc_auc_score(y_train, y_pred_train[:, 1])

    return train_score, test_score, best_thresh, auc, thresholds

def stack_data(rld, reprocess, loaded_loc, processed_loc, data_dir, summarization_int):
    X_ts, X_lab, X_med, X_inf, final_gcs = get_processed_data(loaded_loc, processed_loc,
            rld, reprocess, data_dir, summarization_int)
    num_bins = 24 // summarization_int

    # get individual ts components, switch to numpy
    X_hr = np.vstack(list(X_ts.groupby('patientunitstayid')['hr'].apply(np.hstack).values))
    X_resp = np.vstack(list(X_ts.groupby('patientunitstayid')['resp'].apply(np.hstack).values))
    X_sao2 = np.vstack(list(X_ts.groupby('patientunitstayid')['sao2'].apply(np.hstack).values))
    X_gcs = np.vstack(list(X_ts.groupby('patientunitstayid')['gcs'].apply(np.hstack).values))

    # put them back into ts numpy array
    X_ts = np.hstack([X_hr, X_resp, X_sao2, X_gcs])

    # get other stuff, move to numpy
    X_lab = X_lab.iloc[:, 1:].values
    X_med = X_med.iloc[:, 1:].values
    X_inf = X_inf.iloc[:, 1:].values

    # combine all Xs to do train test split
    # save length of each component array for doing PCA every run
    num_components = [X_ts.shape[1], X_lab.shape[1], X_med.shape[1], X_inf.shape[1]]
    X_stacked = np.hstack([X_ts, X_lab, X_med, X_inf])

    return X_stacked, final_gcs, num_components

def main():
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reload', help="reload data", default=False,
            action='store_true')
    parser.add_argument('-p', '--reprocess', help="reprocess data", default=False,
            action='store_true')
    parser.add_argument('-d', '--data_dir', help='path to data directory', default='data')
    parser.add_argument('-n', '--num_runs', help='number of times to bootstrap', default=10)
    parser.add_argument('-s', '--summarization_int', help='binning interval', default=1)
    args = parser.parse_args()

    # check if loaded/processed dirs exist, check if any files are in them
    existing = os.path.exists(os.path.join(args.data_dir, 'loaded'))
    processed = os.path.exists(os.path.join(args.data_dir, 'loaded', 'processed'))
    results = os.path.exists('model_metrics')

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

    X_stacked, final_gcs, num_components = stack_data(args.reload, args.reprocess,
            loaded_dir, processed_dir, args.data_dir, args.summarization_int)
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
        thresholds_all.append(thresholds[:, None].T)

        # get coefficients
        coeff_all.append(model.coef_)

    # stack arrays and save
    train_scores=np.vstack(train_scores)
    test_scores=np.vstack(test_scores)
    theta_all=np.vstack(theta_all)
    auc_all=np.vstack(auc_all)
    thresholds_all = np.hstack(thresholds_all)

    print("Train Score:", np.mean(train_scores))
    print("Test Score:", np.mean(test_scores))
    print("AUC:", np.mean(auc_all))

    # make experiment dir
    os.makedirs(os.path.join('model_metrics', exp_num))

    np.save(os.path.join(results_dir, exp_num, 'train_scores.npy'), train_scores)
    np.save(os.path.join(results_dir, exp_num, 'test_scores.npy'), test_scores)
    np.save(os.path.join(results_dir, exp_num, 'theta_all.npy'), theta_all)
    np.save(os.path.join(results_dir, exp_num, 'auc_all.npy'), auc_all)
    np.save(os.path.join(results_dir, exp_num, 'thresholds_all.npy'), thresholds_all)

if __name__ == "__main__":
    main()
