"""
setup_model_from_file_evaluateOnly.py
=========================================

Python script that used to evaluate both late arrival and Waalewijn objective
assuming the volunteer distribution is proportional to demand.
"""

import time
from src.instance import Instance
from src.optimization_a_la_evaluate import OptimizationEvaluate
import pandas as pd
import numpy as np


if __name__ == "__main__":

    f = open("data/outputFiles/lastResultsEvaluateOnly.txt", "w")

    # Path to input files
    pop_dir = 'data/inputFiles/population.txt'
    travelA_dir = 'data/inputFiles/travelTimesAmbu-base2loc.txt'
    lambda_dir = 'data/inputFiles/estimate_incident_rate.csv'

    # Read lambda from file
    df_incident = pd.read_csv(lambda_dir)
    estimate_lambda = df_incident['adjusted_estimate'].to_numpy()

    # MEXCLP solution with 25 ambulances (min required for 0.95 coverage within 12 min (10 min driving))
    # and a busy fraction of 0.44 and population as lambda, given current 15 bases
    #115_Apollo_Drive,_Rosedale,_Auckland_0632,_New_Zealand    0
    #2_Shea_Terrace,_Takapuna,_Westlake_0622,_New_Zealand    3
    #47_Pitt_Street,_Auckland_CBD,_Auckland_1010,_New_Zealand    2
    #188_Saint_Heliers_Bay_Road,_St_Heliers,_Auckland_1071,_New_Zealand    2
    #2014/590_Pakuranga_Road,_Bucklands_Beach,_Auckland_2014,_New_Zealand    3
    #23_Atkinson_Avenue,_Otahuhu,_Auckland_1062,_New_Zealand    0
    #32_Sir_William_Avenue,_East_Tamaki,_Auckland_2013,_New_Zealand    1
    #10_Weymouth_Road,_Manurewa,_Auckland_2102,_New_Zealand    1
    #2104/180_Plunket_Avenue,_Manukau,_Auckland_2104,_New_Zealand    3
    #19_Ray_Small_Drive,_Papakura,_Auckland_2110,_New_Zealand    2
    #2_Clinker_Street,_Mount_Roskill,_Auckland_1042,_New_Zealand    2
    #4_Harrison_Road,_Mount_Wellington,_Auckland_1060,_New_Zealand    1
    #117_State_Highway_16,_Whenuapai,_Auckland_0814,_New_Zealand    0
    #75_Wolverton_Street,_Avondale,_Auckland_0600,_New_Zealand    2
    #247_Edmonton_Road,_Te_Atatu_South,_Auckland_0610,_New_Zealand    3

    nBases = 15
    ambus = [0, 3, 2, 2, 3, 0, 1, 1, 3, 2, 2, 1, 0, 2, 3]

    nVolunteers = 100
    volResDelay = 180
    walkingSpeed = 6 #kmh
    ambuBusyProb = 0.44
    ambuResDelay = 180
    threshold = 780 #1km at 6kmh + vol response delay
    maxDispatchDistance = 1.0
    nSteps = 10000

    # Set up problem instances
    inst = Instance(pop_dir, travelA_dir, nVolunteers, volResDelay, walkingSpeed,
                    nBases, ambuResDelay, ambuBusyProb, ambus, threshold,
                    maxDispatchDistance, nSteps)
    optimizerEvaluate = OptimizationEvaluate(inst, estimate_lambda)

    # Assume volunteer distribution proportional to demand
    x_prop = estimate_lambda.copy()
    upper_Waalewijn = optimizerEvaluate.Survival_Waalewijn_Upper()
    nVolunteerScenarios = range(0, 6000 + 1, 100)

    # Evaluating both objectives under different number of volunteers
    f.write('\n\n')
    f.write('Proportional to demand\n')
    f.write('nVolunteers\tLateArrival780\tLateArrival420\tSurvivalWaalewijnModel1\tUpperLimitWaalewijn\tAlpha\n')
    for i in range(len(nVolunteerScenarios)):
        print(nVolunteerScenarios[i])
        optimizerEvaluate.nVolunteers = nVolunteerScenarios[i]

        optimizerEvaluate.threshold = 780
        f.write(str(nVolunteerScenarios[i])+ "\t" +
              str(optimizerEvaluate.evaluateLateArrival_as_in_java(x_prop)) + "\t")
        optimizerEvaluate.threshold = 420
        f.write(str(optimizerEvaluate.evaluateLateArrival_as_in_java(x_prop)) + "\t" +
              str(optimizerEvaluate.evaluateWaalewijn_model1_as_in_java(x_prop)) + "\t" +
              str(upper_Waalewijn) + "\t" +
              str(float(nVolunteerScenarios[i]/11000)) + "\t" +
              "\n")

    f.close()
