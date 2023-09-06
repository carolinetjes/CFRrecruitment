"""
uncertainty_analysis.py
=========================================

Python script that performs uncertainty analysis on profile estimation.
"""
import time
import pandas as pd
import numpy as np
from src.instance import Instance
from src.optimization_profile_time import ProfileOptTime
from scipy.stats import dirichlet

def generate_profile(df, r, num):
    profile_list =[]

    for j in range(num):
        profile = []

        for i in range(df.shape[0]):
            alpha = df.iloc[i].to_numpy()
            alpha += 1e-5
            alpha = alpha * r
            sample = dirichlet.rvs(alpha, size=1)
            profile.append(list(sample[0][:-1]))

        profile = np.array(profile)
        profile_list.append(profile)


    return profile_list


if __name__ == "__main__":


    # Path to input files
    pop_dir = 'data/inputFiles/population.txt'
    travelA_dir = 'data/inputFiles/travelTimesAmbu-base2loc.txt'
    lambda_dir = 'data/inputFiles/estimate_incident_rate.csv'
    profile_dir = 'data/inputFiles/Profiles.xlsx'

    # Read lambda from file
    df_incident = pd.read_csv(lambda_dir)
    estimate_lambda = df_incident['adjusted_estimate'].to_numpy()

    # Read profile from file
    df_profile = pd.read_excel(profile_dir)
    df_profile = df_profile.drop(df_profile.columns[0], axis=1)
    df_profile1 = df_profile.drop(df_profile.columns[-1], axis=1)
    profile_day = df_profile1.to_numpy()

    # Set up problem instances
    nBases = 15
    ambus = [0, 3, 2, 2, 3, 0, 1, 1, 3, 2, 2, 1, 0, 2, 3]

    nVolunteers = 3571
    volResDelay = 180
    walkingSpeed = 6 #kmh
    ambuBusyProb = 0.44
    ambuResDelay = 180
    threshold = 420 #1km at 6kmh + vol response delay
    maxDispatchDistance = 1.0
    nSteps = 10000

    inst = Instance(pop_dir, travelA_dir, nVolunteers, volResDelay, walkingSpeed,
                    nBases, ambuResDelay, ambuBusyProb, ambus, threshold,
                    maxDispatchDistance, nSteps)

    # Additional parameters
    numTimeSeg = 2
    alphaL = [0.17, 0.08]
    lambdaL = [estimate_lambda, estimate_lambda]
    OHCAProbL = [0.7, 0.3]
    profile_night = np.identity(inst.nLocations)
    profileL = [profile_day, profile_night]
    no_profileL = [profile_night, profile_night]

    OptFW = ProfileOptTime(inst, numTimeSeg, alphaL, lambdaL, OHCAProbL, profileL)
    OptFW_no_profile = ProfileOptTime(inst, numTimeSeg, alphaL, lambdaL, OHCAProbL, no_profileL)

    # Proportional and uniform distribution
    x_prop = estimate_lambda.copy()
    x_uni = [x / sum(inst.area) for x in inst.area]


    #--------------------------- Optimize with and without profile-------------

    start = time.time()
    x_waalewijn, _, _ = OptFW.Frankwolfe_LB(x_prop, tol=1e-3, method='w')
    end = time.time()
    print("The time taken to optimize Waalewijn is: ", (end - start)/60, " minutes.\n")

    # Optimize without profile
    start = time.time()
    x_waalewijn_noprofile, _, _ = OptFW_no_profile.Frankwolfe_LB(x_prop, tol=1e-3, method='w')
    end = time.time()
    print("The time taken to optimize Waalewijn is: ", (end - start)/60, " minutes.\n")



    # ------------------------ Uncertainty analysis --------------------------
    profile_list_day = generate_profile(df_profile, 1, 500)

    result_opt = []
    result_uni = []
    result_diag = []
    result_prop = []

    for i in range(100):

        if i%10 == 0:
            print(i)

        profileL = [profile_list_day[i], profile_night]
        OptFW = ProfileOptTime(inst, numTimeSeg, alphaL, lambdaL, OHCAProbL, profileL)

        result_opt.append(OptFW.evaluatePWaalewijn(x_waalewijn))
        result_uni.append(OptFW.evaluatePWaalewijn(x_uni))
        result_diag.append(OptFW.evaluatePWaalewijn(x_waalewijn_noprofile))
        result_prop.append(OptFW.evaluatePWaalewijn(x_prop))


    # Output result to csv file
    df_result = pd.DataFrame()
    df_result['areaunit'] = df_incident['areaunit']
    df_result['opt_profile'] = x_waalewijn
    df_result['opt_diag'] = x_waalewijn_noprofile
    df_result.to_csv('data/outputFiles/uncertainty_result.csv', index=False)


    df_result2 = pd.DataFrame()
    df_result2['opt_waalewijn'] = result_opt
    df_result2['uni'] = result_uni
    df_result2['opt_diag'] = result_diag
    df_result2['propDemand'] = result_prop
    df_result2.to_csv('data/outputFiles/uncertainty_result_evaluation.csv', index=False)
