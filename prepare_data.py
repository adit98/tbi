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

    # drop data after 24 hrs
    hr = hr[hr['offset'] < 24]
    resp = resp[resp['offset'] < 24]
    sao2 = sao2[sao2['offset'] < 24]

    # drop nan rows
    hr = hr.dropna()
    resp = resp.dropna()
    sao2 = sao2.dropna()

    return hr, resp, sao2

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
    lab_data.rename({'labresultoffset': 'offset'})

    return lab_data

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
    replace_dict = {'Home': 6, 'Skilled Nursing Facility': 5, 'Nursing Home': 5,
            'Rehabilitation': 5, 'Other External': 5, 'Floor': 4, 'Other Internal': 4,
            'Step-Down Unit (SDU)': 3, 'Other Hospital': 3, 'ICU': 2, 'Other ICU': 2,
            'Other ICU (CABG)': 2, 'Acute Care/Floor': 2, 'Telemetry': 1, 'Operating Room': 1,
            'Death': 0, 'Other': np.nan}

    discharge_location = demographic_all.loc[:, ['patientunitstayid', 'unitdischargeoffset',
        'unitdischargelocation']]
    discharge_location['unitdischargelocation'].replace(replace_dict, inplace=True)
    discharge_location.drop_duplicates()

    return died, demographic, discharge_location

def process_ts(hr, resp, sao2, gcs, summarization_int=1):
    # put all the ts data into the same dataframe
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

    return ts_data

def process_lab(lab_data):
    # pivot so that each column is a lab
    lab_data_piv = pd.pivot_table(lab_data, index=['patientunitstayid',
        'labresultoffset'], columns='labname', values = 'labresult').reset_index()
    lab_data_piv = lab_data_piv.drop(columns=['labresultoffset'])

    # get counts of each lab for each patient
    lab_cts = lab_data_piv.groupby('patientunitstayid').count().reset_index()

    # get avg lab value for each patient
    lab_avgs = lab_data_piv.groupby('patientunitstayid').mean().reset_index()

    # drop labs that <80% of patients get
    lab_cts = lab_cts[lab_cts.columns[lab_cts.sum() > lab_cts.shape[0] * 4 / 5]]
    lab_avgs = lab_avgs[lab_cts.columns[lab_cts.sum() > lab_cts.shape[0] * 4 / 5]]

    # fill patients missing labs with 0s
    lab_cts = lab_cts.groupby('patientunitstayid').apply(lambda x: x.fillna(0))

    #TODO check this assumption
    # fill patients' missing lab values with avg value of that lab
    lab_avgs = lab_avgs.apply(lambda x: x.fillna(x.median()),
            axis=1).drop(columns=['patientunitstayid'])
    lab_data = pd.concat([lab_cts, lab_avgs], axis=1)

    return lab_data

def process_infusion(inf_data):
    # helper function to get longest word in infusion drug name (avoid duplicates)
    def get_long_boy(df):
        df = df.apply(lambda x: x.strip().split())
        def list_argmax(l):
            return l.index(max(l))

        longest_word = df.apply(list_argmax).values
        for ind in range(len(df)):
            df.iloc[ind] = df.iloc[ind][longest_word[ind]]
        return df

    # modify infusion names (fix typos and dupes)
    inf_data = inf_data[['patientunitstayid', 'drugname', 'drugrate']].drop_duplicates()
    inf_data['drugname'] = inf_data['drugname'].str.replace("-", " ")
    inf_data['drugname'] = inf_data['drugname'].str.replace("(", " ")
    inf_data['drugname'] = inf_data['drugname'].str.replace(")", " ")
    inf_data['drugname'] = inf_data['drugname'].str.replace("/", " ")
    inf_data['drugname'] = inf_data['drugname'].str.lower()
    inf_data['drugname'] = inf_data['drugname'].str.replace("nacl", "sodium")
    inf_data['drugname'] = inf_data['drugname'].str.replace("na", "sodium")
    inf_data['drugname'] = inf_data['drugname'].str.replace("sodiumnograms", "sodium")
    inf_data['drugname'] = inf_data['drugname'].str.replace("without", "")
    inf_data['drugname'] = inf_data['drugname'].str.replace("volume", "")
    inf_data['drugname'] = inf_data['drugname'].str.replace("products", "")
    inf_data['drugname'] = inf_data['drugname'].str.replace("nanograms", "")
    inf_data['drugname'] = inf_data['drugname'].str.replace("units", "")
    inf_data['drugname'] = inf_data['drugname'].str.replace("premix", "")
    inf_data['drugname'] = inf_data['drugname'].str.replace("unknown", "other")
    inf_data['drugname'] = inf_data['drugname'].str.replace("other", "")
    inf_data['drugname'] = inf_data['drugname'].str.replace("platelet", "platelets")
    inf_data['drugname'] = inf_data['drugname'].str.replace("plateletss", "platelets")
    inf_data['drugname'] = inf_data['drugname'].str.replace("prbcs", "prbc")
    inf_data['drugname'] = inf_data['drugname'].str.replace("transfuse", "")
    inf_data['drugname'] = inf_data['drugname'].str.replace("transfusion", "")
    inf_data['drugname'] = inf_data['drugname'].str.replace("tranexamic", "")
    inf_data['drugname'] = inf_data['drugname'].str.replace("transexamic", "")
    inf_data['drugname'] = inf_data['drugname'].str.replace("transeximic", "")
    inf_data['drugname'] = inf_data['drugname'].str.replace("tranxemic", "")
    inf_data['drugname'] = inf_data['drugname'].str.replace("versed", "midazolam")
    inf_data['drugname'] = inf_data['drugname'].str.replace("pca", "patientcontrolledanesthesia")
    inf_data['drugname'] = inf_data['drugname'].str.replace("pcea", "patientcontrolledanesthesia")
    inf_data['drugname'] = inf_data['drugname'].str.replace("sublimaze", "fentanyl")
    inf_data['drugname'] = inf_data['drugname'].str.replace("vasoactives", "vasopressin")
    inf_data['drugname'] = inf_data['drugname'].str.replace("tazo", "zosyn")
    inf_data['drugname'] = inf_data['drugname'].str.replace("zosyn", "piperacillin")
    inf_data['drugname'] = inf_data['drugname'].str.replace("precedex", "dexmedetomidine")
    inf_data['drugname'] = inf_data['drugname'].str.replace("pprbc", "redbloodcell")
    inf_data['drugname'] = inf_data['drugname'].str.replace("prbc", "redbloodcell")
    inf_data['drugname'] = inf_data['drugname'].str.replace("rbc", "redbloodcell")

    # apply helper function
    inf_data['drugname'] = get_long_boy(inf_data['drugname'])

    # drop nan rows and set values to 1 or 0
    inf_data = inf_data.dropna()
    inf_data['drugrate'] = 1

    # pivot infusion names to columns
    inf_data = pd.pivot_table(inf_data, index=['patientunitstayid'], columns=['drugname'], values=['drugrate']).reset_index()

    # get rid of multiindex
    inf_data.columns = inf_data.columns.droplevel(0)
    inf_data = inf_data.rename(columns={'': 'patientunitstayid'})

    # drop extraneous columns
    inf_data = inf_data.drop(columns=['w', 'with', 'mg', 'ml', 'mcg', 'plt', 'min', 'mvi', 'ns', 'nss', 'std', 'hr', 'kg', 'tf', 'tko'])
    inf_data = inf_data.fillna(0)

    # keep columns where more than 1% of patients got infusion
    inf_data = inf_data[inf_data.columns[inf_data.mean() > 0.01]]

    return inf_data

def process_medication(med_data):
    # select relevant columns
    med_data = med_data[['patientunitstayid', 'dosage', 'gtc']].dropna().drop_duplicates()

    # set values to 0/1
    med_data['dosage'] = 1

    # pivot lab names
    med_data = pd.pivot_table(med_data, index=['patientunitstayid'], columns='gtc')
    med_data = med_data.fillna(0).reset_index()

    # get rid of multiindex
    med_data.columns = med_data.columns.droplevel(0)
    med_data = med_data.rename(columns={'': 'patientunitstayid'})

    # keep columns where more than 20% of patients got medication
    med_data = med_data[med_data.columns[med_data.mean() > 0.2]]

    return med_data

'''Wrapper for getting processed data, if we need to reload we do so here'''
def get_processed_data(loaded_loc, processed_loc, rld, reprocess, data_dir, summarization_int=1):
    # load data from individual component files if available
    if not rld:
        hr = pd.read_csv(os.path.join(loaded_loc, 'hr.csv'))
        resp = pd.read_csv(os.path.join(loaded_loc, 'resp.csv'))
        sao2 = pd.read_csv(os.path.join(loaded_loc, 'sao2.csv'))
        gcs = pd.read_csv(os.path.join(loaded_loc, 'gcs.csv'))
        final_gcs = pd.read_csv(os.path.join(loaded_loc, 'final_gcs.csv'))
        lab_data = pd.read_csv(os.path.join(loaded_loc, 'lab_data.csv'))
        mort_data = pd.read_csv(os.path.join(loaded_loc, 'mort_data.csv'))
        dem_data = pd.read_csv(os.path.join(loaded_loc, 'dem_data.csv'))
        discharge_data = pd.read_csv(os.path.join(loaded_loc, 'discharge_data.csv'))

    else:
        hr, resp, sao2 = get_physiology(data_dir, 'alpaca_hr.csv',
                'alpaca_resp.csv', 'alpaca_sao2.csv')
        gcs, final_gcs = get_motor_gcs(os.path.join(data_dir, 'patient_motor.csv'))
        lab_data = get_lab_data(os.path.join(data_dir, 'lab_data.csv'))
        mort_data, dem_data, discharge_data = get_demographics(os.path.join(data_dir,
            'patient_demographics_data.csv'))

        # save data that we split into individual components
        hr.to_csv(os.path.join(loaded_loc, 'hr.csv'))
        resp.to_csv(os.path.join(loaded_loc, 'resp.csv'))
        sao2.to_csv(os.path.join(loaded_loc, 'sao2.csv'))
        gcs.to_csv(os.path.join(loaded_loc, 'gcs.csv'))
        final_gcs.to_csv(os.path.join(loaded_loc, 'final_gcs.csv'))
        lab_data.to_csv(os.path.join(loaded_loc, 'lab_data.csv'))
        mort_data.to_csv(os.path.join(loaded_loc, 'mort_data.csv'))
        dem_data.to_csv(os.path.join(loaded_loc, 'dem_data.csv'))
        discharge_data.to_csv(os.path.join(loaded_loc, 'discharge_data.csv'))

    if reprocess:
        # load raw med and infusion data
        medication_data = pd.read_csv(os.path.join(data_dir, 'medication_data_long_query.csv'),
                low_memory=False)
        infusion_data = pd.read_csv(os.path.join(data_dir, 'infusion_data_long_query.csv'),
                low_memory=False)

        # create dataframe to hold all time series data
        ts_data = process_ts(hr, resp, sao2, gcs)

        # process and reduce dimensionality of infusion and medication data
        # TODO figure out if this step can be placed where we do the second fillna
        medication_data = process_medication(medication_data)
        infusion_data = process_infusion(infusion_data)
        lab_data = process_lab(lab_data)

        # get patient list - we include all patients from ts_data
        # TODO in future, remove patients with <24 hrs using demographic data
        patient_list = ts_data[['patientunitstayid']].drop_duplicates().reset_index()

        # order the rest of the data by the patient order in patient list
        ts_data = patient_list.merge(ts_data, how='left')
        lab_data = patient_list.merge(lab_data, how='left')
        medication_data = patient_list.merge(medication_data, how='left')
        infusion_data = patient_list.merge(infusion_data, how='left')

        # fill the rest of the patients indicator variables with 0s
        medication_data = medication_data.fillna(0)

        # fill the rest of the patients indicator variables with 0s
        infusion_data = infusion_data.fillna(0)

        # fill the rest of the patients lab data
        lab_cts = lab_data.iloc[:, :lab_data.shape[0]//2]
        lab_cts = lab_cts.groupby('patientunitstayid').apply(lambda x: x.fillna(0))
        lab_avgs = lab_data.iloc[:, lab_data.shape[0]//2:]
        lab_avgs = lab_avgs.apply(lambda x: x.fillna(x.median()), axis=1)
        lab_data = pd.concat([lab_cts, lab_avgs], axis=1)

        ts_data.to_csv(os.path.join(processed_loc, 'ts_processed.csv'))
        lab_data.to_csv(os.path.join(processed_loc, 'lab_processed.csv'))
        medication_data.to_csv(os.path.join(processed_loc, 'medication_processed.csv'))
        infusion_data.to_csv(os.path.join(processed_loc, 'infusion_processed.csv'))

    else:
        ts_data = pd.read_csv(os.path.join(processed_loc, 'ts_processed.csv'))
        lab_data = pd.read_csv(os.path.join(processed_loc, 'lab_processed.csv'))
        medication_data = pd.read_csv(os.path.join(processed_loc, 'medication_processed.csv'))
        infusion_data = pd.read_csv(os.path.join(processed_loc, 'infusion_processed.csv'))

        # get patient list and put into dataframe
        patient_list = pd.DataFrame(columns=['patientunitstayid'])
        patient_list['patientunitstayid'] = ts_data.patientunitstayid.unique()

    # finally, order gcs
    final_gcs = patient_list.merge(final_gcs, how='left')

    # drop unnamed columns
    ts_data = ts_data.loc[:, ~ts_data.columns.str.contains('^Unnamed')]
    lab_data = lab_data.loc[:, ~lab_data.columns.str.contains('^Unnamed')]
    infusion_data = infusion_data.loc[:, ~infusion_data.columns.str.contains('^Unnamed')]
    medication_data = medication_data.loc[:, ~medication_data.columns.astype(str).str.contains('^Unnamed')]
    final_gcs = final_gcs.loc[:, ~final_gcs.columns.str.contains('^Unnamed')]

    # drop offset columns
    ts_data = ts_data.loc[:, ~ts_data.columns.str.contains('^offset_bin')]
    lab_data = lab_data.loc[:, ~lab_data.columns.str.contains('^offset_bin')]
    medication_data = medication_data.loc[:, ~medication_data.columns.astype(str).str.contains('^offset_bin')]
    infusion_data = infusion_data.loc[:, ~infusion_data.columns.str.contains('^offset_bin')]
    final_gcs = final_gcs.loc[:, ~final_gcs.columns.str.contains('^offset_bin')]

    return ts_data, lab_data, medication_data, infusion_data, final_gcs
