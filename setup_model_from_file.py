"""
setup_model_from_file.py
=========================================

Python script that optimize both late arrival and Waalewijn survival objectives
without profiles using the Greedy algorithm under different number of
volunteers, and evaluate the performance of optimal solutions.
"""
import time
from src.instance import Instance
from src.optimization_a_la_evaluate import OptimizationEvaluate
import pandas as pd
import numpy as np
import sys


if __name__ == "__main__":

    f = open("data/outputFiles/lastResultsNoProfiles.txt", "w")
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
    threshold = 420 #1km at 6kmh + vol response delay
    maxDispatchDistance = 1.0
    nSteps = 10000

    # Set up problem instances
    inst = Instance(pop_dir, travelA_dir, nVolunteers, volResDelay, walkingSpeed,
                    nBases, ambuResDelay, ambuBusyProb, ambus, threshold,
                    maxDispatchDistance, nSteps)
    optimizerEvaluate = OptimizationEvaluate(inst, estimate_lambda)

    # Set up proportional and uniform distribution
    x_prop = estimate_lambda.copy()
    x_uni = [x / sum(inst.area) for x in inst.area]
    nVolunteerScenarios = range(0, 5000+1, 100)

    print('Area: ', optimizerEvaluate.area)
    print('Proportional: ', x_prop)
    print('Uniform: ', x_uni)


    # Optimize both objectives using greedy algorithm

    x_greedy_waalewijn_model1 = [ [ 0 for i in range(inst.nLocations)] for j in range(len(nVolunteerScenarios)) ]
    x_greedy_late420 = [ [ 0 for i in range(inst.nLocations)] for j in range(len(nVolunteerScenarios)) ]

    for i in range(len(nVolunteerScenarios)):
        optimizerEvaluate.nVolunteers = nVolunteerScenarios[i]
        print(optimizerEvaluate.nVolunteers, 'volunteers and response time target', optimizerEvaluate.threshold)

        print('Optimizing for Waalewijn survival with', nVolunteerScenarios[i], 'volunteers.')
        start = time.time()
        x_greedy_waalewijn_model1[i] = optimizerEvaluate.Greedy_Waalewijn_model1_N(nVolunteerScenarios[i])
        end = time.time()
        print("The time taken to optimize Waalewijn is: ", end - start)

        print('Optimizing for Late arrival (greedy) with', nVolunteerScenarios[i], 'volunteers and a threshold of 420.')
        start = time.time()
        x_greedy_late420[i] = optimizerEvaluate.Greedy_lateArrival()
        end = time.time()
        print("The time taken to optimize late arrival with a threshold of 420 is: ", end - start)

    # Output to file
    f.write('\n')
    f.write('Area: ' +  str(optimizerEvaluate.area) + '\n')
    f.write('Lambda: ' + str(optimizerEvaluate.lambdaL))
    f.write('\n')
    f.write('Ambus: ' +  str(ambus) + '\n')
    f.write('\n \n')

    headerlineX  = ""
    for i in range(len(nVolunteerScenarios)):
        headerlineX += 'x_waalewijn_model1_'+ str(nVolunteerScenarios[i]) + '\t'
        headerlineX += 'x_lateArrival420_greedy_'+ str(nVolunteerScenarios[i]) + '\t'
    headerlineX += ('\n')

    f.write(headerlineX)
    for j in range(inst.nLocations): #each row is an area unit
        for i in range(len(nVolunteerScenarios)):
            f.write( str(x_greedy_waalewijn_model1[i][j]) + '\t')
            f.write( str(x_greedy_late420[i][j]) + '\t')
        f.write('\n')
    f.write('\n')


    # Evaluate optimal solutions and prop, uni distribution
    f.write('\n')
    f.write('Waalewijn_model1\n')

    headers_with_ambu = 'nVolunteers\tSurvWaalewijn_model1GreedyOpt\tLateArrivalOpt_greedy\tUniform\tProp\n'
    f.write(headers_with_ambu)
    for i in range(len(nVolunteerScenarios)):
        optimizerEvaluate.nVolunteers = nVolunteerScenarios[i]

        f.write(str(nVolunteerScenarios[i])+ "\t" +
              str(optimizerEvaluate.evaluateWaalewijn_model1_as_in_java(x_greedy_waalewijn_model1[i])) + "\t" +
              str(optimizerEvaluate.evaluateWaalewijn_model1_as_in_java(x_greedy_late420[i])) + "\t" +
              str(optimizerEvaluate.evaluateWaalewijn_model1_as_in_java(x_uni)) + "\t" +
              str(optimizerEvaluate.evaluateWaalewijn_model1_as_in_java(x_prop)) + "\t" + "\n")



    f.write('\n')
    optimizerEvaluate.threshold = 420
    f.write('LateArrivals (threshold 420 sec)\n')
    f.write(headers_with_ambu)
    for i in range(len(nVolunteerScenarios)):
        optimizerEvaluate.nVolunteers = nVolunteerScenarios[i]

        f.write(str(nVolunteerScenarios[i])+ "\t" +
              str(optimizerEvaluate.evaluateLateArrival_as_in_java(x_greedy_waalewijn_model1[i])) + "\t" +
              str(optimizerEvaluate.evaluateLateArrival_as_in_java(x_greedy_late420[i])) + "\t" +
              str(optimizerEvaluate.evaluateLateArrival_as_in_java(x_uni)) + "\t" +
              str(optimizerEvaluate.evaluateLateArrival_as_in_java(x_prop)) + "\t" + "\n")

    f.close()
