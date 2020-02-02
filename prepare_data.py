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
    lab = pd.read_csv(lab_filename)
    lab = lab[lab['labresultoffset'] > -1]
    lab = lab[lab['labresultoffset'] <= 24*60]
    labtypes = list(lab['labname'].drop_duplicates())

    # Getting table of averages over first 24 hours
    l = labtypes[0]
    labtypedata = lab[lab['labname'] == l]
    labtypedata_avgs = labtypedata.groupby('patientunitstayid').mean()['labresult']
    labtypedata_avgs = labtypedata_avgs.to_frame().reset_index().rename(columns = {'labresult': l})
    for l in labtypes[1:]:
        labtypedata = lab[lab['labname'] == l]
        labtypedata_avgs1 = labtypedata.groupby('patientunitstayid').mean()['labresult']
        labtypedata_avgs1 = labtypedata_avgs1.to_frame().reset_index().rename(columns = {'labresult': l})
        labtypedata_avgs = labtypedata_avgs.merge(labtypedata_avgs1, how='outer', on='patientunitstayid')

    labtypedata_avgs = labtypedata_avgs[labtypedata_avgs.columns[labtypedata_avgs.isnull().mean() < 0.2]]

    lab_avgs_map = {}
    for l in labtypes:
        if l in labtypedata_avgs.columns:
            col = list(labtypedata_avgs[l].dropna())
            if len(col) != 0:
                avg = sum(col)/len(col)
                lab_avgs_map[l] = avg

            else:
                labtypedata_avgs = labtypedata_avgs.drop(columns=l)

    for l in labtypedata_avgs.columns:
        if l != 'patientunitstayid':
            col = labtypedata_avgs[l]
            labtypedata_avgs[l] = col.fillna(lab_avgs_map[l])

    l = labtypes[0]
    labtypedata = lab[lab['labname'] == l]
    labtypedata_cts = labtypedata.groupby('patientunitstayid').count()['labresult']
    labtypedata_cts = labtypedata_cts.to_frame().reset_index().rename(columns = {'labresult': l})

    for l in labtypes[1:]:
        labtypedata = lab[lab['labname'] == l]
        labtypedata_cts1 = labtypedata.groupby('patientunitstayid').count()['labresult']
        labtypedata_cts1 = labtypedata_cts1.to_frame().reset_index().rename(columns = {'labresult': l})
        labtypedata_cts = labtypedata_cts.merge(labtypedata_cts1, how='outer', on='patientunitstayid')

    l = labtypes[0]
    labtypedata = lab[lab['labname'] == l]
    labtypedata_cts = labtypedata.groupby('patientunitstayid').count()['labresult']
    labtypedata_cts = labtypedata_cts.to_frame().reset_index().rename(columns = {'labresult': l})

    for l in labtypes[1:]:
        labtypedata = lab[lab['labname'] == l]
        labtypedata_cts1 = labtypedata.groupby('patientunitstayid').count()['labresult']
        labtypedata_cts1 = labtypedata_cts1.to_frame().reset_index().rename(columns = {'labresult':l})
        labtypedata_cts = labtypedata_cts.merge(labtypedata_cts1, how='outer', on='patientunitstayid')

    labtypedata_cts = labtypedata_cts[labtypedata_cts.columns[labtypedata_cts.isnull().mean() < 0.2]]
    labtypedata_cts = labtypedata_cts.apply(lambda x: x.fillna(x.median()),
            axis=0) > labtypedata_cts.median()

    return labtypedata_cts, labtypedata_avgs

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

    return demographic

def bin_data(summarization_int):
    pass


