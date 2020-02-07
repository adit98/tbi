# IMPORTS
import pandas as pd
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

    # drop nan rows (no need to interpolate here if we are interpolating when binning)
    hr = hr.dropna()
    resp = resp.dropna()
    sao2 = sao2.dropna()

    return hr, resp, sao2

def get_motor_gcs(gcs_filename):
    gcs = pd.read_csv(gcs_filename)
    gcs = gcs[gcs['observationoffset'] > -1]
    gcs['observationoffset'] = gcs['observationoffset'] / 60
    final_gcs_id = gcs.groupby('patientunitstayid')['observationoffset'].transform(max) == gcs['observationoffset']

    final_gcs = gcs[final_gcs_id]

    return gcs, final_gcs

def get_lab_data(lab_filename):
    # Loading lab data
    lab = pd.read_csv('data/lab_data.csv')
    lab = lab.loc[lab['labresultoffset'] > -1]
    labtypes = list(lab['labname'].drop_duplicates())

    labmeasures = measurements.copy()
    toAdd = pd.DataFrame(columns=['patientunitstayid','key','value', 'offset'])
    toAdd['patientunitstayid'] = lab['patientunitstayid']
    toAdd['key'] = lab['labname']
    toAdd['value'] = lab['labresult']
    toAdd['offset'] = lab['labresultoffset']
    labmeasures = labmeasures.merge(toAdd, how = 'outer')
    labmeasures = labmeasures.dropna()

    return labmeasures

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

    discharge_location = demographic_all.loc[:, ['patientunitstayid',
        'unitdischargeoffset', 'unitdischargelocation']]
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

def rein(df):
    df = df.set_index('offsetBin')
    df = df.drop(['patientunitstayid', 'key'], axis = 1)
    return df.reindex(np.arange(24)).fillna(method = 'ffill').fillna(method = 'bfill')

def bin_data(summarization_int, hr, resp, sao2, lab_data, med_data, inf_data):
    #TODO add medication and infusion data
    measurements = pd.DataFrame(columns=['patientunitstayid', 'key', 'value', 'offset'])
    hr['key'] = 'hr'
    resp['key'] = 'resp'
    sao2['key'] = 'sao2'
    measurements = measurements.merge(hr, how = 'outer').merge(resp, how = 'outer').merge(sao2, how = 'outer').merge(gcs, how = 'outer')

    measurements = measurements.dropna()

    m = lab_data.copy()
    #m = m[m['patientunitstayid'] < 200000]
    m = m[m['offset'] < 1440]
    m['offsetBin'] = (m['offset']/summarization_int).astype(int)
    m = m.drop('offset', axis = 1)
    measures_grouped = m.groupby(['patientunitstayid', 'offsetBin', 'key']).mean().reset_index()
    m2 = measures_grouped.copy()

    #TODO use parallel processing here to speed up
    m2 = m2.groupby(['patientunitstayid', 'key']).apply(rein)

    m2 = m2.reset_index()
    m2['pat_off'] = m2['patientunitstayid'].astype(str) + '_' + m2['offsetBin'].astype(str)
    m2 = m2.pivot(index='pat_off', columns='key', values='value')

    return m2
