# IMPORTS
import pandas as pd
import numpy as np
import argparse
import os

def get_physiology(data_dir, hr_filename, resp_filename, sao2_filename):
    # load data
    hr = pd.read_csv(os.path.join(data_dir, hr_filename))
    resp = pd.read_csv(os.path.join(data_dir, resp_filename))
    sao2 = pd.read_csv(os.path.join(data_dir, sao2_filename))

    # convert minute offset to hrs
    hr['offset'] = hr['offset'] / 60
    resp['offset'] = resp['offset'] / 60
    sao2['offset'] = sao2['offset'] / 60

    hr = hr[hr['offset'] < 24]
    resp = resp[resp['offset'] < 24]
    sao2 = sao2[sao2['offset'] < 24]

    # drop nan rows
    hr = hr.dropna()
    resp = resp.dropna()
    sao2 = sao2.dropna()

    return hr, resp, sao2

def get_aperiodic(data_dir, systolicbp_filename, diastolicbp_filename, meanbp_filename):
    # load data
    systolic = pd.read_csv(os.path.join(data_dir, systolicbp_filename))
    diastolic = pd.read_csv(os.path.join(data_dir, diastolicbp_filename))
    meanbp = pd.read_csv(os.path.join(data_dir, meanbp_filename))

    # convert minute offset to hrs
    systolic['offset'] = systolic['offset'] / 60
    diastolic['offset'] = diastolic['offset'] / 60
    meanbp['offset'] = meanbp['offset'] / 60

    systolic = systolic[systolic['offset'] < 24]
    diastolic = diastolic[diastolic['offset'] < 24]
    meanbp = meanbp[meanbp['offset'] < 24]

    # drop nan rows
    systolic = systolic.dropna()
    diastolic = diastolic.dropna()
    meanbp = meanbp.dropna()

    return systolic, diastolic, meanbp

def get_nursecharting(data_dir, verbal_filename, eyes_filename, temperature_filename):
    # load data
    verbal = pd.read_csv(os.path.join(data_dir, verbal_filename))
    eyes = pd.read_csv(os.path.join(data_dir, eyes_filename))
    temperature = pd.read_csv(os.path.join(data_dir, temperature_filename))

    # convert minute offset to hrs
    verbal['offset'] = verbal['offset'] / 60
    eyes['offset'] = eyes['offset'] / 60
    temperature['offset'] = temperature['offset'] / 60

    verbal = verbal[verbal['offset'] < 24]
    eyes = eyes[eyes['offset'] < 24]
    temperature = temperature[temperature['offset'] < 24]

    # drop nan rows
    verbal = verbal.dropna()
    eyes = eyes.dropna()
    temperature = temperature.dropna()

    return verbal, eyes, temperature

def get_motor_gcs(gcs_filename):
    # read file
    gcs = pd.read_csv(gcs_filename)

    # drop invalids
    gcs = gcs[gcs['observationoffset'] > -1]

    # convert offset to hours
    gcs['observationoffset'] = gcs['observationoffset'] / 60

    # get final gcs
    final_gcs_id = gcs.groupby('patientunitstayid')['observationoffset'].transform(max) == gcs['observationoffset']
    final_gcs = gcs[final_gcs_id][['patientunitstayid', 'Value']]

    # drop measurements outside first 24 hours
    gcs = gcs[gcs['observationoffset'] < 24]

    # transform gcs to have same labels as other time series data
    gcs = gcs.rename(columns={'Key': 'key', 'Value': 'value', 'observationoffset': 'offset'})

    # drop nan rows
    gcs = gcs.dropna()

    return gcs, final_gcs

def get_lab_data(lab_filename):
    # Loading lab data
    lab_data = pd.read_csv('data/lab_data.csv')

    # drop rows that are irrelevant (< -1) or outside of first 24 hours
    lab_data = lab_data.loc[lab_data['labresultoffset'] > -1]
    lab_data = lab_data.loc[lab_data['labresultoffset'] <= 24*60]

    # extract relevant and rename columns
    lab_data = lab_data[['patientunitstayid', 'labresultoffset', 'labname', 'labresult']]
    lab_data.rename({'labresultoffset':'offset'})

    # pivot so that each column is a lab
    lab_data_piv = pd.pivot_table(lab_data, index=['patientunitstayid',
        'labresultoffset'], columns='labname', values = 'labresult').reset_index()

    # TODO exclude missing lab features after selecting the correct patient set

    # get counts of each lab for each patient
    lab_counts = lab_data_piv.groupby('patientunitstayid').count().reset_index()
    lab_counts = lab_counts.drop(columns=['labresultoffset'])

    # get avg lab value for each patient
    lab_avgs = lab_data_piv.groupby('patientunitstayid').mean().reset_index()
    lab_avgs = lab_avgs.drop(columns=['labresultoffset'])

    # drop labs that <80% of patients get
    lab_counts = lab_counts[lab_counts.columns[lab_counts.sum() > lab_counts.shape[0] * 4 / 5]]
    lab_avgs = lab_avgs[lab_counts.columns[lab_counts.sum() > lab_counts.shape[0] * 4 / 5]]

    return lab_counts, lab_avgs

def get_demographics(dem_filename):
    # Loading demographic data
    demographic_all = pd.read_csv('data/patient_demographics_data.csv')

    died = demographic_all.loc[:, ['patientunitstayid', 'unitdischargestatus']]
    died['alive'] = died['unitdischargestatus'] == 'Alive'
    died = died.drop_duplicates()

    # Keeping the following columns (numerical for now)
    to_keep = ['age', 'admissionheight', 'admissionweight', 'patientunitstayid']
    demographic = demographic_all[to_keep]

    demographic = demographic.replace('> 89', 90)

    # get discharge location (to use as labels)
#    replace_dict = {'Home': 6, 'Skilled Nursing Facility': 5, 'Nursing Home': 5,
#            'Rehabilitation': 5, 'Other External': 5, 'Floor': 4, 'Other Internal': 4,
#            'Step-Down Unit (SDU)': 3, 'Other Hospital': 3, 'ICU': 2, 'Other ICU': 2,
#            'Other ICU (CABG)': 2, 'Acute Care/Floor': 2, 'Telemetry': 1, 'Operating Room': 1,
#            'Death': 0, 'Other': np.nan}
    replace_dict = {'Home': 2, 'Rehabilitation': 1,
            'Skilled Nursing Facility': 0, 'Nursing Home': 0,'Death': 0,
            'Other External': np.nan, 'Floor': np.nan, 'Other Internal': np.nan,
            'Step-Down Unit (SDU)': np.nan, 'Other Hospital': np.nan,
            'ICU': np.nan, 'Other ICU': np.nan, 'Other ICU (CABG)': np.nan,
            'Acute Care/Floor': np.nan, 'Telemetry': np.nan, 'Operating Room': np.nan,
            'Other': np.nan}

    discharge_location = demographic_all.loc[:, ['patientunitstayid', 'unitdischargeoffset',
        'unitdischargelocation']]
    discharge_location['unitdischargelocation'].replace(replace_dict, inplace=True)
    discharge_location.drop_duplicates()

    return died, demographic, discharge_location

def get_medication(med_filename):
    # Loading medication data
    medication_all = pd.read_csv(med_filename)
    medtypes = list(medication_all['drugname'].drop_duplicates().dropna())

    # Getting table of indicator variables for each patient
    m = medtypes[0]
    medtypedata = medication_all[medication_all['drugname'] == m][['patientunitstayid']].drop_duplicates()#, 'frequency']]
    indicator = np.ones((medtypedata.shape[0], 1))
    medtypedata[m] = indicator
    for m in medtypes[1:]:
        medtypedata1 = medication_all[medication_all['drugname'] == m][['patientunitstayid']].drop_duplicates()
        indicator = np.ones((medtypedata1.shape[0], 1))
        medtypedata1[m] = indicator
        medtypedata = medtypedata.merge(medtypedata1, how='outer', on='patientunitstayid')

    medtypedata = medtypedata.fillna(0)

    return medtypedata

def get_infusion(inf_filename):
    # Loading infusion data
    infusion_all = pd.read_csv(inf_filename)
    infusiontypes = list(infusion_all['drugname'].drop_duplicates().dropna())

    # Getting table of indicator variables for each patient
    i = infusiontypes[0]
    infusiontypedata = infusion_all[infusion_all['drugname'] == i][['patientunitstayid']].drop_duplicates()
    indicator = np.ones((infusiontypedata.shape[0], 1))
    infusiontypedata[i] = indicator

    for i in infusiontypes[1:]:
        infusiontypedata1 = infusion_all[infusion_all['drugname'] == i][['patientunitstayid']].drop_duplicates()
        indicator = np.ones((infusiontypedata1.shape[0], 1))
        infusiontypedata1[i] = indicator
        infusiontypedata = infusiontypedata.merge(infusiontypedata1, how='outer', on='patientunitstayid')

    infusiontypedata = infusiontypedata.fillna(0)

    return infusiontypedata

'''Wrapper for getting processed data, if we need to reload we do so here'''
def get_processed_data(summarization_int, processed_loc, rld, rebin, data_dir):
    # populate all the dataframes
    if not rld:
        hr = pd.read_csv(os.path.join(processed_loc, 'hr.csv'))
        resp = pd.read_csv(os.path.join(processed_loc, 'resp.csv'))
        sao2 = pd.read_csv(os.path.join(processed_loc, 'sao2.csv'))
        gcs = pd.read_csv(os.path.join(processed_loc, 'gcs.csv'))
        final_gcs = pd.read_csv(os.path.join(processed_loc, 'final_gcs.csv'))
        lab_data_cts = pd.read_csv(os.path.join(processed_loc, 'lab_data_cts.csv'))
        lab_data_avgs = pd.read_csv(os.path.join(processed_loc, 'lab_data_avgs.csv'))
        mort_data = pd.read_csv(os.path.join(processed_loc, 'mort_data.csv'))
        dem_data = pd.read_csv(os.path.join(processed_loc, 'dem_data.csv'))
        discharge_data = pd.read_csv(os.path.join(processed_loc, 'discharge_data.csv'))
        medication_data = pd.read_csv(os.path.join(processed_loc,
            'medication_data.csv'))
        infusion_data = pd.read_csv(os.path.join(processed_loc,
            'infusion_data.csv'))

    else:
        hr, resp, sao2 = get_physiology(data_dir, 'alpaca_hr.csv',
                'alpaca_resp.csv', 'alpaca_sao2.csv')
        gcs, final_gcs = get_motor_gcs(os.path.join(data_dir, 'patient_motor.csv'))
        lab_data_cts, lab_data_avgs = get_lab_data(os.path.join(data_dir, 'lab_data.csv'))
        mort_data, dem_data, discharge_data = get_demographics(os.path.join(data_dir,
            'patient_demographics_data.csv'))
        medication_data = get_medication(os.path.join(data_dir,
            'medication_data_long_query.csv'))
        infusion_data = get_infusion(os.path.join(data_dir,
            'infusion_data_long_query.csv'))

        # save all the dataframes
        hr.to_csv(os.path.join(processed_loc, 'hr.csv'))
        resp.to_csv(os.path.join(processed_loc, 'resp.csv'))
        sao2.to_csv(os.path.join(processed_loc, 'sao2.csv'))
        gcs.to_csv(os.path.join(processed_loc, 'gcs.csv'))
        final_gcs.to_csv(os.path.join(processed_loc, 'final_gcs.csv'))
        lab_data_cts.to_csv(os.path.join(processed_loc, 'lab_data_cts.csv'))
        lab_data_avgs.to_csv(os.path.join(processed_loc, 'lab_data_avgs.csv'))
        mort_data.to_csv(os.path.join(processed_loc, 'mort_data.csv'))
        dem_data.to_csv(os.path.join(processed_loc, 'dem_data.csv'))
        discharge_data.to_csv(os.path.join(processed_loc, 'discharge_data.csv'))
        medication_data.to_csv(os.path.join(processed_loc, 'medication_data.csv'))
        infusion_data.to_csv(os.path.join(processed_loc, 'infusion_data.csv'))

    if rebin:
        # create dataframe to hold all time series data
        ts_data = pd.DataFrame(columns=['patientunitstayid', 'key', 'value', 'offset'])
        hr['key'] = 'hr'
        resp['key'] = 'resp'
        sao2['key'] = 'sao2'
        gcs['key'] = 'gcs'
        ts_data = ts_data.merge(hr, how = 'outer').merge(resp, how = 'outer').merge(sao2,
                how = 'outer').merge(gcs, how = 'outer')

        # drop extra columns, nan rows
        ts_data = ts_data.drop(columns=['origin']).dropna()

        # calculate bins, drop offset, calculate bin avg
        ts_data['offset_bin'] = ts_data['offset'] // summarization_int
        ts_data = ts_data.drop(columns=['offset'])
        ts_data = ts_data.groupby(['patientunitstayid', 'offset_bin', 'key']).mean().reset_index()

        # function to reindex dataframe (ensure we have all bins)
        def rein(df):
            num_bins = 24 // summarization_int
            if num_bins != 24:
                print(num_bins, summarization_int)
            df = df.set_index('offset_bin')
            df = df.drop(['patientunitstayid', 'key'], axis = 1)
            return df.reindex(index=np.arange(num_bins).astype(int))

        # set bin as index and reindex
        ts_data = ts_data.loc[:, ~ts_data.columns.str.contains('^Unnamed')]
        ts_data = ts_data.groupby(['patientunitstayid', 'key']).apply(rein).reset_index()

        # fill missing bins, drop patients with no measurements for any of the features
        ts_data = ts_data.groupby(['patientunitstayid', 'key']).apply(lambda x: x.fillna(method='bfill').fillna(method='ffill')).reset_index()
        ts_data = ts_data.pivot_table(index=['patientunitstayid', 'offset_bin'],
                columns='key', values='value').dropna().reset_index()

        # get patient list - requires getting common patients across all df
        ts_patients = ts_data[['patientunitstayid']]
        lab_patients = lab_data_cts[['patientunitstayid']]
        medication_patients = medication_data[['patientunitstayid']]
        infusion_patients = infusion_data[['patientunitstayid']]

        patient_list = ts_data[['patientunitstayid']].drop_duplicates().reset_index()

        # order the rest of the data by the patient order in patient list
        ts_data = patient_list.merge(ts_data, how='left')
        lab_data_cts = patient_list.merge(lab_data_cts, how='left')
        lab_data_avgs = patient_list.merge(lab_data_avgs, how='left')
        medication_data = patient_list.merge(medication_data, how='left')
        infusion_data = patient_list.merge(infusion_data, how='left')

        # now fill nan in lab data, fill avgs with median, cts with 0
        lab_data_cts = lab_data_cts.groupby('patientunitstayid').apply(lambda x: x.fillna(0))
        lab_data_avgs = lab_data_avgs.apply(lambda x: x.fillna(x.median()), axis=1)
        lab_data = pd.concat([lab_data_cts, lab_data_avgs], axis=1)

        # now fill nan in medication data
        medication_data = medication_data.fillna(0)

        # now fill nan in infusion data
        infusion_data = infusion_data.fillna(0)

        ts_data.to_csv(os.path.join(data_dir, 'processed', 'binned', 'ts_binned.csv'))
        lab_data.to_csv(os.path.join(data_dir, 'processed', 'binned', 'lab_binned.csv'))
        medication_data.to_csv(os.path.join(data_dir, 'processed', 'binned',
            'medication_binned.csv'))
        infusion_data.to_csv(os.path.join(data_dir, 'processed', 'binned',
            'infusion_binned.csv'))

    else:
        ts_data = pd.read_csv(os.path.join(data_dir, 'processed', 'binned', 'ts_binned.csv'))
        lab_data = pd.read_csv(os.path.join(data_dir, 'processed', 'binned', 'lab_binned.csv'))
        medication_data = pd.read_csv(os.path.join(data_dir, 'processed', 'binned',
            'medication_binned.csv'))
        infusion_data = pd.read_csv(os.path.join(data_dir, 'processed', 'binned',
            'infusion_binned.csv'))

        # get patient list and put into dataframe
        patient_list = pd.DataFrame(columns=['patientunitstayid'])
        patient_list['patientunitstayid'] = ts_data.patientunitstayid.unique()

    # finally, order gcs
    final_gcs = patient_list.merge(final_gcs, how='left')

    return ts_data, lab_data, medication_data, infusion_data, final_gcs
