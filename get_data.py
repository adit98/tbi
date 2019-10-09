import pandas as pd
import psycopg2 as psql
import os

PGHOST='10.162.38.59'
PGDATABASE='eicu'
PGUSER='eicu'
PGPASSWORD='eicu'

conn_string = "host="+ PGHOST +" port=" + "5432" +" dbname=" \
        + PGDATABASE + " user=" + PGUSER + " password="+ PGPASSWORD
conn = psql.connect(conn_string)
print("Connected!")
cursor = conn.cursor()

queries = {'physiology_query' : "SELECT * \
    FROM eicu_crd.apacheApsVar INNER JOIN eicu_crd.diagnosis \
    ON eicu_crd.apacheApsVar.patientunitstayid = eicu_crd.diagnosis.patientunitstayid \
    WHERE (SUBSTRING(diagnosisstring, 1,25) = 'burns/trauma|trauma - CNS' \
    OR SUBSTRING(diagnosisstring, 1,23) = 'neurologic|trauma - CNS') \
    AND diagnosisstring NOT LIKE '%spinal cord%' LIMIT {} OFFSET {};",

'apache_pred_query' : "SELECT * \
    FROM eicu_crd.apachePredVar INNER JOIN eicu_crd.diagnosis \
    ON eicu_crd.apachePredVar.patientunitstayid = eicu_crd.diagnosis.patientunitstayid \
    WHERE (SUBSTRING(diagnosisstring, 1,25) = 'burns/trauma|trauma - CNS' \
    OR SUBSTRING(diagnosisstring, 1,23) = 'neurologic|trauma - CNS') \
    AND diagnosisstring NOT LIKE '%spinal cord%' LIMIT {} OFFSET {};",

'infusion_query' : "SELECT * \
    FROM eicu_crd.infusionDrug INNER JOIN eicu_crd.diagnosis \
    ON eicu_crd.infusionDrug.patientunitstayid = eicu_crd.diagnosis.patientunitstayid \
    WHERE (SUBSTRING(diagnosisstring, 1,25) = 'burns/trauma|trauma - CNS' \
    OR SUBSTRING(diagnosisstring, 1,23) = 'neurologic|trauma - CNS') \
    AND diagnosisstring NOT LIKE '%spinal cord%' LIMIT {} OFFSET {};",

'nurse_charting_query' : "SELECT * \
    FROM eicu_crd.diagnosis \
    INNER JOIN eicu_crd.nursecharting \
    ON eicu_crd.diagnosis.patientunitstayid = eicu_crd.nursecharting.patientunitstayid \
    WHERE nursingchartcelltypevallabel = 'Glasgow coma score' \
    AND (SUBSTRING(diagnosisstring, 1,25) = 'burns/trauma|trauma - CNS' \
    OR SUBSTRING(diagnosisstring, 1,23) = 'neurologic|trauma - CNS') \
    AND diagnosisstring not like '%spinal cord%' LIMIT {} OFFSET {};",

'medication_query' : "SELECT * \
    FROM eicu_crd.medication INNER JOIN eicu_crd.diagnosis \
    ON eicu_crd.medication.patientunitstayid = eicu_crd.diagnosis.patientunitstayid \
    WHERE (SUBSTRING(diagnosisstring, 1,25) = 'burns/trauma|trauma - CNS' \
    OR SUBSTRING(diagnosisstring, 1,23) = 'neurologic|trauma - CNS') \
    AND diagnosisstring NOT LIKE '%spinal cord%' LIMIT {} OFFSET {};",

'patient_demographics_query' : "SELECT * \
    FROM eicu_crd.patient INNER JOIN eicu_crd.diagnosis \
    ON eicu_crd.patient.patientunitstayid = eicu_crd.diagnosis.patientunitstayid \
    WHERE (SUBSTRING(diagnosisstring, 1,25) = 'burns/trauma|trauma - CNS' \
    OR SUBSTRING(diagnosisstring, 1,23) = 'neurologic|trauma - CNS') \
    AND diagnosisstring NOT LIKE '%spinal cord%' LIMIT {} OFFSET {};",

'respiratory_query' : "SELECT * \
    FROM eicu_crd.respiratoryCharting INNER JOIN eicu_crd.diagnosis \
    ON eicu_crd.respiratoryCharting.patientunitstayid = eicu_crd.diagnosis.patientunitstayid \
    WHERE (SUBSTRING(diagnosisstring, 1,25) = 'burns/trauma|trauma - CNS' \
    OR SUBSTRING(diagnosisstring, 1,23) = 'neurologic|trauma - CNS') \
    AND diagnosisstring NOT LIKE '%spinal cord%' LIMIT {} OFFSET {};",

'aperiodic_vitals_query' : "SELECT * \
    FROM eicu_crd.vitalAperiodic INNER JOIN eicu_crd.diagnosis \
    ON eicu_crd.vitalAperiodic.patientunitstayid = eicu_crd.diagnosis.patientunitstayid \
    WHERE (SUBSTRING(diagnosisstring, 1,25) = 'burns/trauma|trauma - CNS' \
    OR SUBSTRING(diagnosisstring, 1,23) = 'neurologic|trauma - CNS') \
    AND diagnosisstring NOT LIKE '%spinal cord%' LIMIT {} OFFSET {};",

'periodic_vitals_query' : "SELECT * \
    FROM eicu_crd.vitalPeriodic INNER JOIN eicu_crd.diagnosis \
    ON eicu_crd.vitalPeriodic.patientunitstayid = eicu_crd.diagnosis.patientunitstayid \
    WHERE (SUBSTRING(diagnosisstring, 1,25) = 'burns/trauma|trauma - CNS' \
    OR SUBSTRING(diagnosisstring, 1,23) = 'neurologic|trauma - CNS') \
    AND diagnosisstring NOT LIKE '%spinal cord%' LIMIT {} OFFSET {};"}

for key in queries.keys():
    offset = 0
    window_size = 1000000
    filename = key[:-5] + 'data.csv'
    df = None

    while True:
        query = queries[key].format(window_size, offset)
        if df is not None:
            cursor.execute(query)
            data = cursor.fetchall()
            if not data:
                break

            print("got data", len(data), "existing data", len(df))
            df = df.append(pd.DataFrame(data, columns=df.columns))

        else:
            df = pd.read_sql(query, conn)

        offset += window_size

    if not os.path.exists("data"):
        os.makedirs("data")

    df.to_csv(os.path.join("data", filename))
    print("got 1")
