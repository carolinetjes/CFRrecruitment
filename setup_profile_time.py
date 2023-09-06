"""
setup_profile_time.py
=========================================

Python script that optimize both late arrival and Waalewijn survival objectives
with time profiles using the FW algorithm under different number of
volunteers, and evaluate the performance of optimal solutions.
"""
import time
import pandas as pd
import numpy as np
from src.instance import Instance
from src.optimization_profile_time import ProfileOptTime

def str_every_entry_on_new_line(my_array):
    """
    Generate string with each element in the array takes one line.

    Parameters
    ----------
    my_array : list
        An array that we want to print.

    Returns
    -------
    result : str
        String with each element separated by a new line.
    """
    result = ""

    for a in my_array:
        result +="\n"
        result += str(a)
    return result


if __name__ == "__main__":

    f = open("data/outputFiles/lastResultsWithTimeProfiles.txt", "w")
    f_no_profile = open("data/outputFiles/lastResultsWithoutTimeProfiles.txt", "w")

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
    df_profile = df_profile.drop(df_profile.columns[-1], axis=1)
    profile_day = df_profile.to_numpy()

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



    #----------------------------- Solve Optimization problem ---------------------

    nVolunteerScenarios = range(1000, 30000 + 1, 1000)

    profile_waalewijn = []
    profile_late420 = []

    for n in nVolunteerScenarios:
        OptFW.nVolunteers = n

        print('Optimizing for Waalewijn survival with', n, 'volunteers with profile.')
        start = time.time()
        x_waalewijn, _, _ = OptFW.Frankwolfe_LB(x_prop, tol=3e-3, verbose=True, method='w')
        profile_waalewijn.append(x_waalewijn)
        end = time.time()
        print("The time taken to optimize Waalewijn is: ", (end - start)/60, " minutes.\n")

        print('Optimizing for Late arrival 420 with', n, 'volunteers with profile.')
        start = time.time()
        x_late420, _, _ = OptFW.Frankwolfe_LB(x_prop, tol=1e-3, verbose=False, method='l')
        profile_late420.append(x_late420)
        end = time.time()
        print("The time taken to optimize late arrival 420 is: ", (end - start)/60, " minutes.\n")

    no_profile_waalewijn = []
    no_profile_late420 = []

    for n in nVolunteerScenarios:
        OptFW_no_profile.nVolunteers = n

        print('Optimizing for Waalewijn survival with', n, 'volunteers without profile.')
        start = time.time()
        x_waalewijn, _, _ = OptFW_no_profile.Frankwolfe_LB(x_prop, tol=3e-3, verbose=True, method='w')
        no_profile_waalewijn.append(x_waalewijn)
        end = time.time()
        print("The time taken to optimize Waalewijn is: ", (end - start)/60, " minutes.\n")

        print('Optimizing for Late arrival 420 with', n, 'volunteers without profile.')
        start = time.time()
        x_late420, _, _ = OptFW_no_profile.Frankwolfe_LB(x_prop, tol=1e-3, verbose=False, method='l')
        no_profile_late420.append(x_late420)
        end = time.time()
        print("The time taken to optimize late arrival 420 is: ", (end - start)/60, " minutes.\n")

    #----------------------------- Write optimal x and nu ---------------------


    headerlineX  = ""
    for i in range(len(nVolunteerScenarios)):
        headerlineX += 'x_waalewijn_profile_'+ str(nVolunteerScenarios[i]) + '\t'
        headerlineX += 'x_lateArr420_profile_'+ str(nVolunteerScenarios[i]) + '\t'
    headerlineX += ('\n')

    f.write(headerlineX)
    for j in range(inst.nLocations): #each row is an area unit
        for i in range(len(nVolunteerScenarios)):
            f.write( str(profile_waalewijn[i][j]) + '\t')
            f.write( str(profile_late420[i][j]) + '\t')
        f.write('\n')
    f.write('\n\n')


    nu_waalewijn_L = [profile_day.T @ x for x in profile_waalewijn]
    nu_late420_L = [profile_day.T @ x for x in profile_late420]

    headerlineNu  = ""
    for i in range(len(nVolunteerScenarios)):
        headerlineNu += 'nu_waalewijn_profile_day_'+ str(nVolunteerScenarios[i]) + '\t'
        headerlineNu += 'nu_lateArr420_profile_day_'+ str(nVolunteerScenarios[i]) + '\t'
    headerlineNu += ('\n')

    f.write(headerlineNu)
    for j in range(inst.nLocations): #each row is an area unit
        for i in range(len(nVolunteerScenarios)):
            f.write( str(nu_waalewijn_L[i][j]) + '\t')
            f.write( str(nu_late420_L[i][j]) + '\t')
        f.write('\n')
    f.write('\n\n')




    headerlineX_no_profile  = ""
    for i in range(len(nVolunteerScenarios)):
        headerlineX_no_profile += 'x_waalewijn_no_profile_'+ str(nVolunteerScenarios[i]) + '\t'
        headerlineX_no_profile += 'x_lateArr420_no_profile_'+ str(nVolunteerScenarios[i]) + '\t'
    headerlineX_no_profile += ('\n')

    f_no_profile.write(headerlineX_no_profile)
    for j in range(inst.nLocations): #each row is an area unit
        for i in range(len(nVolunteerScenarios)):
            f_no_profile.write( str(no_profile_waalewijn[i][j]) + '\t')
            f_no_profile.write( str(no_profile_late420[i][j]) + '\t')
        f_no_profile.write('\n')
    f_no_profile.write('\n\n')


    #----------------------------- Evaluation ---------------------


    title_waalewijn = 'nVolunteers\tSurvWaalewijnOpt\tSurvWaalewijnOpt_noProfile\tSurvUniform\tSurvProp\n'
    title_late420 = 'nVolunteers\tlateArrival420Opt\tlateArrival420_Opt_noProfile\tlateArrival420Uniform\tlateArrival420Prop\n'

    nVolunteerScenarios = [0] + list(nVolunteerScenarios)
    profile_waalewijn = [x_prop] + profile_waalewijn
    profile_late420 = [x_prop] + profile_late420
    no_profile_waalewijn = [x_prop] + no_profile_waalewijn
    no_profile_late420 = [x_prop] + no_profile_late420

    f.write('\n\n')
    OptFW.threshold = 420
    f.write('With Profiles:\n')
    f.write('LateArrivals (threshold '+ str(420) +' sec)\n')
    f.write(title_late420)
    for i in range(len(nVolunteerScenarios)):
        OptFW.nVolunteers = nVolunteerScenarios[i]

        s = str(nVolunteerScenarios[i])+ "\t"
        s += str(OptFW.pLateArrival(profile_late420[i])) + "\t"
        s += str(OptFW.pLateArrival(no_profile_late420[i])) + "\t"
        s += str(OptFW.pLateArrival(x_uni)) + "\t"
        s += str(OptFW.pLateArrival(x_prop)) + "\t" + "\n"

        f.write(s)

    f.write('\n\n')
    f.write('Waalewijn\n')
    f.write(title_waalewijn)
    for i in range(len(nVolunteerScenarios)):
        OptFW.nVolunteers = nVolunteerScenarios[i]

        s = str(nVolunteerScenarios[i])+ "\t"
        s += str(OptFW.evaluatePWaalewijn(profile_waalewijn[i])) + "\t"
        s += str(OptFW.evaluatePWaalewijn(no_profile_waalewijn[i])) + "\t"
        s += str(OptFW.evaluatePWaalewijn(x_uni)) + "\t"
        s += str(OptFW.evaluatePWaalewijn(x_prop)) + "\t" + "\n"

        f.write(s)

    f.close()

    f_no_profile.write('\n\n')
    OptFW.threshold = 420
    f_no_profile.write('Without Profiles:\n')
    f_no_profile.write('LateArrivals (threshold '+ str(420) +' sec)\n')
    f_no_profile.write(title_late420)
    for i in range(len(nVolunteerScenarios)):
        OptFW_no_profile.nVolunteers = nVolunteerScenarios[i]

        s = str(nVolunteerScenarios[i])+ "\t"
        s += str(OptFW_no_profile.pLateArrival(profile_late420[i])) + "\t"
        s += str(OptFW_no_profile.pLateArrival(no_profile_late420[i])) + "\t"
        s += str(OptFW_no_profile.pLateArrival(x_uni)) + "\t"
        s += str(OptFW_no_profile.pLateArrival(x_prop)) + "\t" + "\n"

        f_no_profile.write(s)

    f_no_profile.write('\n\n')
    f_no_profile.write('Waalewijn\n')
    f_no_profile.write(title_waalewijn)
    for i in range(len(nVolunteerScenarios)):
        OptFW_no_profile.nVolunteers = nVolunteerScenarios[i]

        s = str(nVolunteerScenarios[i])+ "\t"
        s += str(OptFW_no_profile.evaluatePWaalewijn(profile_waalewijn[i])) + "\t"
        s += str(OptFW_no_profile.evaluatePWaalewijn(no_profile_waalewijn[i])) + "\t"
        s += str(OptFW_no_profile.evaluatePWaalewijn(x_uni)) + "\t"
        s += str(OptFW_no_profile.evaluatePWaalewijn(x_prop)) + "\t" + "\n"

        f_no_profile.write(s)

    f_no_profile.close()

    #------------------ Write Optimal solutions to csv ---------------------

    x = profile_waalewijn[1:] + profile_late420[1:]
    x_header = []

    for i in range(len(nVolunteerScenarios)-1):
        x_header.append('x_waalewijn_profile_'+ str(nVolunteerScenarios[i+1]))

    for i in range(len(nVolunteerScenarios)-1):
        x_header.append('x_lateArr420_profile_'+ str(nVolunteerScenarios[i+1]))

    df = pd.DataFrame(x)
    df = df.transpose()
    df.columns = x_header
    df.to_csv('data/outputFiles/ResultsWithTimeProfiles_x.csv', index=False)



    x_no_profile = no_profile_waalewijn[1:] + no_profile_late420[1:]
    x_header_no_profile = []

    for i in range(len(nVolunteerScenarios)-1):
        x_header_no_profile.append('x_waalewijn_no_profile_'+ str(nVolunteerScenarios[i+1]))

    for i in range(len(nVolunteerScenarios)-1):
        x_header_no_profile.append('x_lateArr420_no_profile_'+ str(nVolunteerScenarios[i+1]))

    df_no_profile = pd.DataFrame(x_no_profile)
    df_no_profile = df_no_profile.transpose()
    df_no_profile.columns = x_header_no_profile
    df_no_profile.to_csv('data/outputFiles/ResultsWithoutTimeProfiles_x.csv', index=False)
