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

#'infusion_query' : "with p_id as ( \
#        select distinct patientunitstayid, diagnosisstring \
#        from eicu_crd.diagnosis \
#        WHERE diagnosisstring LIKE '%trauma - CNS|intracranial%' \
#    ) \
#    SELECT * \
#    FROM p_id INNER JOIN eicu_crd.infusionDrug \
#    ON eicu_crd.infusionDrug.patientunitstayid = p_id.patientunitstayid \
#    LIMIT {} OFFSET {};",


queries = {
    'nurse_charting_query' : "with p_id as ( \
            select distinct patientunitstayid, diagnosisstring \
            from eicu_crd.diagnosis \
            WHERE diagnosisstring LIKE '%trauma - CNS|intracranial%' \
        ) \
        SELECT p_id.patientunitstayid, nursingchartoffset, \
        nursingchartentryoffset, nursingchartcelltypevallabel, \
        nursingchartcelltypevalname, nursingchartvalue \
        FROM p_id INNER JOIN eicu_crd.nursecharting \
        ON p_id.patientunitstayid = eicu_crd.nursecharting.patientunitstayid \
        WHERE nursingchartcelltypevallabel = 'Glasgow coma score' \
        LIMIT {} OFFSET {};",

    'medication_query' : "with p_id as ( \
            select distinct patientunitstayid, diagnosisstring \
            from eicu_crd.diagnosis \
            WHERE diagnosisstring LIKE '%trauma - CNS|intracranial%' \
        ) \
        SELECT * \
        FROM p_id INNER JOIN eicu_crd.medication \
        ON eicu_crd.medication.patientunitstayid = p_id.patientunitstayid \
        LIMIT {} OFFSET {};",

    'patient_demographics_query' : "with p_id as ( \
            select distinct patientunitstayid, diagnosisstring \
            from eicu_crd.diagnosis \
            WHERE diagnosisstring LIKE '%trauma - CNS|intracranial%' \
        ) \
        SELECT * \
        FROM p_id INNER JOIN eicu_crd.patient \
        ON eicu_crd.patient.patientunitstayid = p_id.patientunitstayid \
        LIMIT {} OFFSET {};",

    'respiratoryCharting_query' : "with p_id as ( \
            select distinct patientunitstayid, diagnosisstring \
            from eicu_crd.diagnosis \
            WHERE diagnosisstring LIKE '%trauma - CNS|intracranial%' \
        ) \
        SELECT * \
        FROM p_id INNER JOIN eicu_crd.respiratoryCharting \
        ON eicu_crd.respiratoryCharting.patientunitstayid = p_id.patientunitstayid \
        LIMIT {} OFFSET {};", \

    'aperiodic_vitals_query' : "with p_id as ( \
            select distinct patientunitstayid, diagnosisstring \
            from eicu_crd.diagnosis \
            WHERE diagnosisstring LIKE '%trauma - CNS|intracranial%' \
        ) \
        SELECT * \
        FROM p_id INNER JOIN eicu_crd.vitalAperiodic \
        ON eicu_crd.vitalAperiodic.patientunitstayid = p_id.patientunitstayid \
        LIMIT {} OFFSET {};",

    'periodic_vitals_query' : "with p_id as ( \
            select distinct patientunitstayid, diagnosisstring \
            from eicu_crd.diagnosis \
            WHERE diagnosisstring LIKE '%trauma - CNS|intracranial%' \
        ) \
        SELECT * \
        FROM p_id INNER JOIN eicu_crd.vitalPeriodic \
        ON eicu_crd.vitalPeriodic.patientunitstayid = p_id.patientunitstayid \
        LIMIT {} OFFSET {};"}

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
