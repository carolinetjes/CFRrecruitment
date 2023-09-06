from mysql.connector import connect, Error
from credentials import host, port, user, pswd, name
import pandas as pd
import numpy as np
import random

if __name__ == "__main__":

    #-------------------- Read GoodSAM Data -------------------------
    try:
        with connect(
            host=host,
            port=port,
            database=name,
            user=user,
            password=pswd
        ) as database, \
            database.cursor() as cursor:
            query_col = "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'goodsam'"
            query = "SELECT * FROM goodsam WHERE Time_CallEnteredQueue >= '2017-12-01T00:00:00.000'"

            cursor.execute(query_col)
            result_col = cursor.fetchall()
            column_name = [x[0] for x in result_col]

            cursor.execute(query)
            result = cursor.fetchall()
            df = pd.DataFrame(result, columns=column_name)

    except Error as e:
        print(e)

    #-------------------- Read GoodSAM Data -------------------------
    try:
        f = open('data/inputFiles/Meshblock data.txt', 'r')
    except:
        message = "This input file '{} does not exist.".format(pop)
        sys.exit(message)

    L = [line.split() for line in f][4:]
    meshblock = [int(node[1]) for node in L]

    #------------------- Extract Auckland Data -----------------------
    df_auckland = df[df['MB_IncidentAdd'].isin(meshblock)].reset_index(drop=True)
    incidents = list(set(df_auckland['Master_Incident_Number']))


    # --------------- Extract Incidents with alerts for sampling response -----
    df_alert = df_auckland[df_auckland['responderCallStatus'] != ''].reset_index(drop=True)


    # --------------- Generate Synthetic Dataset ------------------------
    synthetic_data = pd.DataFrame(columns = list(df.columns))
    columns = list(df.columns)

    # Sample 50 incidents
    sample_incidents = random.sample(incidents, 50)
    index = 0
    for incident in sample_incidents:
        df_incident = df_auckland[df_auckland['Master_Incident_Number'] == incident].reset_index(drop=True)
        mb = int(df_incident.iloc[0]['MB_IncidentAdd'])

        # '' suggests that this incident has no alert sending out
        if '' in list(df_incident['responderCallStatus']):
            nAlert = 0
        else:
            nAlert = len(df_incident)

        # Add random noise to the number of alerts sending out
        random_noise = random.sample([-1, 1], 1)[0]
        nAlert = max(0, nAlert + random_noise)

        sample_alert = [None] * len(columns)

        if nAlert == 0:
            sample_alert[columns.index('Master_Incident_Number')] = index
            sample_alert[columns.index('MB_IncidentAdd')] = mb
            sample_alert[columns.index('responderCallStatus')] = ''
            synthetic_data.loc[len(synthetic_data)] = sample_alert

            index += 1
            continue

        for i in range(nAlert):
            alert = df_alert.sample(1)

            sample_alert[columns.index('Master_Incident_Number')] = index
            sample_alert[columns.index('MB_IncidentAdd')] = mb
            sample_alert[columns.index('responderCallStatus')] = list(alert['responderCallStatus'])[0]

            synthetic_data.loc[len(synthetic_data)] = sample_alert

        index += 1

    synthetic_data.to_csv('data/Auckland data/synthetic_goodsam_data.csv', index=False)
