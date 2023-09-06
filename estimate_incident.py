"""
estimate_incident.py
=========================================

Python script that estimate the incident rate following Dicker et. al (2019).
"""

from mysql.connector import connect, Error
from credentials import host, port, user, pswd, name
import pandas as pd
import numpy as np

if __name__ == "__main__":

    input_dir = 'data/inputFiles/Input regression incident rate.xlsx'

    #-------------------- Read four predictors--------------------------------
    df_input = pd.read_excel(input_dir)

    # Compute baseline average for each predictor
    mean_maori = np.mean(df_input['Proportion Maori'])
    mean_pacific = np.mean(df_input['Proportion Pacific'])
    mean_gt65 = np.mean(df_input['Proportion 65+'])
    mean_male = np.mean(df_input['Proportion Male'])

    # Compute deviation from baseline average for each area unit
    maori_increase = (df_input['Proportion Maori'].to_numpy() - mean_maori) * 100
    pacific_increase = (df_input['Proportion Pacific'].to_numpy() - mean_pacific) * 100
    gt65_increase = (df_input['Proportion 65+'].to_numpy() - mean_gt65) * 100
    male_increase = (df_input['Proportion Male'].to_numpy() - mean_male) * 100

    # Constant from Dicker et. al (2019)
    const_maori = 0.01
    const_pacific = 0.006
    const_gt65 = 0.037
    const_male = 0.037


    # ------------------ Read GoodSAM Data ----------------------

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
            query = "SELECT * FROM goodsam"

            cursor.execute(query_col)
            result_col = cursor.fetchall()
            column_name = [x[0] for x in result_col]

            cursor.execute(query)
            result = cursor.fetchall()
            df = pd.DataFrame(result, columns=column_name)

    except Error as e:
        print(e)

    # ------------------ Extract Auckland Data ----------------------

    meshblock_dir = 'data/inputFiles/Meshblock data.txt'
    areaunit_dir = 'data/inputFiles/population.txt'

    # Match meshblock to areaunit
    try:
        f = open(meshblock_dir, 'r')
    except:
        message = "This input file '{} does not exist.".format(pop)
        sys.exit(message)

    L = [line.split() for line in f][4:]
    meshblock = [int(node[1]) for node in L]

    Meshblock_to_Areaunit = {}
    for node in L:
        Meshblock_to_Areaunit[int(node[1])] = int(node[4])

    f.close()

    # Get index of areaunit, population and area
    try:
        f = open(areaunit_dir, 'r')
    except:
        message = "This input file '{} does not exist.".format(pop)
        sys.exit(message)

    L = [line.split() for line in f][1:]

    areaunit_index = {}
    for i in range(len(L)):
        node = L[i]
        areaunit_index[int(node[0])] = i

    population = np.array([int(node[1]) for node in L])
    areaL = np.array([float(node[2]) for node in L])
    f.close()

    # Extract only Auckland data
    df_auckland = df[df['MB_IncidentAdd'].isin(meshblock)].reset_index(drop=True)
    incidents = set(df_auckland['Master_Incident_Number'])


    # ---------------- Compute empirical rate -----------------------------

    # Count number of incidents for each areaunit
    total_incidents = np.zeros(287)

    for incident in incidents:
        df_incident = df_auckland[df_auckland['Master_Incident_Number'] == incident].reset_index(drop=True)
        m = int(df_incident.iloc[0]['MB_IncidentAdd'])
        au = Meshblock_to_Areaunit[m]
        au_index = areaunit_index[au]
        total_incidents[au_index] += 1

    # Compute empirical rate and density
    empirical = total_incidents / sum(total_incidents)
    empirical_density = empirical / areaL

    # -------------- Compute estimate rate, Dicker et al. (2019) -------------

    # Input regression incident rate.xlsx has a different ordering of area unit
    # So we want to reoder the data so that the ordering matches

    areaunit_order = [list(df_input['Area Unit']).index(x) for x in areaunit_index.keys()]
    maori_increase = maori_increase[areaunit_order]
    pacific_increase = pacific_increase[areaunit_order]
    gt65_increase = gt65_increase[areaunit_order]
    male_increase = male_increase[areaunit_order]

    # Compute estimate rate and density
    incident_rate = np.ones(df_input.shape[0])
    estimate = (incident_rate + (maori_increase * const_maori) + (pacific_increase * const_pacific) + (gt65_increase * const_gt65) + (male_increase * const_male)) * population
    estimate_rate = estimate / sum(estimate)
    estimate_density = estimate_rate / areaL

    #-------------- Adjust CBD and airport -----------------------------------
    # Compute baseline ohca rate
    ohca_rate = total_incidents / population
    baseline_rate = np.mean(ohca_rate)
    estimate_count = baseline_rate * estimate

    # Adjust count for CBD and airport
    CBD_airport_index = [areaunit_index[514102], areaunit_index[514103], areaunit_index[524200]]
    adjusted_estimate_count = estimate_count.copy()
    adjusted_estimate_count[CBD_airport_index] = total_incidents[CBD_airport_index]

    adjusted_estimate_rate = adjusted_estimate_count / sum(adjusted_estimate_count)
    adjusted_estimate_density = adjusted_estimate_rate / areaL

    # -------------------- Output to csv file ------------------------------
    df_count = pd.DataFrame()
    df_count['areaunit'] = list(areaunit_index.keys())
    df_count['adjust_estimate_count'] = adjusted_estimate_count
    df_count['empirical_count'] = total_incidents
    df_count['estimate_count'] = estimate_count
    df_count.to_csv('data/outputFiles/empirical_vs_estimate_count.csv', index=False)

    df_rate = pd.DataFrame()
    df_rate['areaunit'] = list(areaunit_index.keys())
    df_rate['empirical'] = empirical
    df_rate['estimate'] = estimate_rate
    df_rate['adjusted_estimate'] = adjusted_estimate_rate
    df_rate['empirical_density'] = estimate_density
    df_rate['estimate_density'] = estimate_density
    df_rate['adjusted_estimate_density'] = adjusted_estimate_density
    df_rate.to_csv('data/inputFiles/estimate_incident_rate.csv', index=False)
