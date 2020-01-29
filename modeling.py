# IMPORTS
import pandas as pd
import argparse
import os
from prepare_data import *

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reload', help="reload data", default=False,
            action='store_true')
    parser.add_argument('-d', '--data_dir', help='path to data directory', default='data')
    args = parser.parse_args()

    # check if processed dir exists
    existing = os.path.exists(os.path.join(args.data_dir, 'processed'))
    if not existing:
        # create dir
        os.makedirs(os.path.join(args.data_dir, 'processed'))
        assert args.reload, "Must pass the -r flag, no processed data exists"

    processed_loc = os.path.join(args.data_dir, 'processed')

    if args.reload:
        hr, resp, sao2 = get_physiology(args.data_dir, 'alpaca_hr.csv', 'alpaca_resp.csv', 'alpaca_sao2.csv')
        gcs, final_gcs = get_motor_gcs(os.path.join(args.data_dir, 'patient_motor.csv'))
        lab_data_cts, lab_data_avgs = get_lab_data(os.path.join(args.data_dir, 'lab_data.csv'))
        dem_data = get_demographics(os.path.join(args.data_dir, 'patient_demographics_data.csv'))

        # save all the dataframes
        hr.to_csv(os.path.join(processed_loc, 'hr.csv'))
        resp.to_csv(os.path.join(processed_loc, 'resp.csv'))
        sao2.to_csv(os.path.join(processed_loc, 'sao2.csv'))
        gcs.to_csv(os.path.join(processed_loc, 'gcs.csv'))
        final_gcs.to_csv(os.path.join(processed_loc, 'final_gcs.csv'))
        lab_data_cts.to_csv(os.path.join(processed_loc, 'lab_data_cts.csv'))
        lab_data_avgs.to_csv(os.path.join(processed_loc, 'lab_data_avgs.csv'))
        dem_data.to_csv(os.path.join(processed_loc, 'dem_data.csv'))

    else:
        # populate all the dataframes
        hr = pd.read_csv(os.path.join(processed_loc, 'hr.csv'))
        resp = pd.read_csv(os.path.join(processed_loc, 'resp.csv'))
        sao2 = pd.read_csv(os.path.join(processed_loc, 'sao2.csv'))
        gcs = pd.read_csv(os.path.join(processed_loc, 'gcs.csv'))
        final_gcs = pd.read_csv(os.path.join(processed_loc, 'final_gcs.csv'))
        lab_data_cts = pd.read_csv(os.path.join(processed_loc, 'lab_data_cts.csv'))
        lab_data_avgs = pd.read_csv(os.path.join(processed_loc, 'lab_data_avgs.csv'))
        dem_data = pd.read_csv(os.path.join(processed_loc, 'dem_data.csv'))
