"""
optimization_a_la_evaluate.py
=========================================

Module that used to compute ideal volunteer **location** distribution using
greedy algorithm without considering profile.
"""


import numpy as np
import math
import time
from .instance import Instance

class OptimizationEvaluate:
    """
    Compute ideal volunteer distribution with differnt evaluation functions

    Args
    ----
    inst : Instance
        An Instance object that contains information read from the Input files.
    lamb : list
        Input OHCA demand distribution.

    Attributes
    ----------
    nVolunteers : int
        Number of volunteers.
    volResDelay : float
        Volunteer response time delay in seconds from patient collapse.
    walkingSpeed : float
        Volunteer constance walking speed.
    nBases : int
        Number of ambulance bases.
    ambuResDelay : float
        Ambulance response time delay in seconds from patient collapse.
    busyProb : float
        Probability that an ambulance will be busy at time of the call.
    ambus : list
        Ambulance distribution over the set of bases.
    threshold : float
        Response time target.
    maxDispatchDistance : float
        Dispatch radius in kilometer within which volunteers will be alerted.
    nSteps : int
        Number of steps in greedy algorithm.
    nLocations : int
        Number of locations.
    population : list
        List of population for each region.
    area : list
        List of area for each region.
    travelTime_A : list
        Travel time for ambulance from bases to each location.
    lambdaL : list
        OHCA demand distribution.
    nAmbulances : int
        Number of ambulances.
    maxDispatchTime : int
        Max dispatch radius in time by assuming constant walking speed for
        volunteers.
    waalewijnSurvivalMatrix : numpy.array
        Pre-caculated waalewijn survival with different CPR and EMS time.
    """

    def __init__(self, inst: Instance, lamb: list) -> None:

        # Read from input problem instance
        self.nVolunteers = inst.nVolunteers
        self.volResDelay = inst.volResDelay
        self.walkingSpeed = inst.volWalkingSpeed
        self.nBases = inst.nBases
        self.ambuResDelay = inst.ambuResDelay
        self.busyProb = inst.ambuBusyProb
        self.ambus = inst.ambuDist
        self.threshold = inst.threshold
        self.maxDispatchDistance = inst.maxDispatchDistance
        self.nSteps = inst.nSteps
        self.nLocations = inst.nLocations
        self.area = inst.area
        self.population = inst.population
        self.travelTime_A = inst.travelTime_A

        # Set demand distribution
        self.lambdaL = lamb

        # Additional parameters
        self.nAmbulances = sum(self.ambus)
        self.maxDispatchTime = int((self.maxDispatchDistance/self.walkingSpeed)*3600 + self.volResDelay)
        self.waalewijnSurvivalMatrix = self.calcWaalewijnSurvivalMatrix()

    #----------------------------------------------------------------
    #       Greedy late arrival
    #----------------------------------------------------------------

    '''
    always compute late arrivals WITHOUT ambus
    '''
    def LateArrivalInLocation(self, nu: float, loc: int) -> float:
        """
        Compute late arrival rate for a location.

        Parameters
        ----------
        nu : float
            Probability that volunteers appear in given location.
        loc : int
            Index of location.

        Returns
        -------
        float
            Probability that a response time is greater than $\tau$.
        """
        nAmbuWithinTarget = 0

        for i in range(0,self.nBases):
            if (self.ambus[i] > 0 and self.travelTime_A[i][loc] + self.ambuResDelay <= self.threshold) :
                nAmbuWithinTarget += self.ambus[i]

        pNoAmbuAvailable = self.busyProb**nAmbuWithinTarget

        distance_tau = self.walkingSpeed/3600 * (self.threshold - self.volResDelay) # in kms
        exponent = - self.nVolunteers * math.pi * distance_tau * distance_tau / self.area[loc]
        pNoVolunteerAvailable = math.exp(exponent*nu)

        return self.lambdaL[loc] * pNoAmbuAvailable * pNoVolunteerAvailable

    def AdditionalOnTimeArrivalInLocation(self, nu: float, loc: int, stepsize: float) -> float:
        """
        Compute additional late arrival probability when a small volunteer
        mass is added to the given location.

        Parameters
        ----------
        nu : float
            Original probability that volunteers appear in given location.
        loc : int
            Index of location.
        stepsize : float
            Additional volunteer mass added.

        Returns
        -------
        float
            Difference in late arrival probability.
        """

        # discretized time to seconds. In continuous form this should be the derivative of f_i with respect to nu_i
        newNu = nu + stepsize

        return self.LateArrivalInLocation(nu, loc) - self.LateArrivalInLocation(newNu, loc)

    def Greedy_lateArrival_exact(self) -> list:
        """
        Compute volunteer distrubution by optimizing late arrival rate
        for each location using greedy algorithm.

        Returns
        -------
        nu : list
            Optimized volunteer distribution.
        """

        nu = [0.0] * self.nLocations
        if (sum(self.ambus) > 0 or self.nVolunteers == 0):
            print('Exact method for minimizing late arrivals not yet implemented for case with ambulance.')
            return nu

        locationWithHigestLambdaPerArea = [0.0] * self.nLocations
        lambdaPerArea = [0.0] * self.nLocations
        for i in range(len(lambdaPerArea)):

            lambdaPerArea[i] = self.lambdaL[i]/self.area[i] #cj: python can divide whole arrays by doing np.divide(lambdaL, area)
        lambdaPerArea_temp = np.asarray(lambdaPerArea)

        for i in range(len(lambdaPerArea)):
            j = lambdaPerArea_temp.argmax()

            locationWithHigestLambdaPerArea[i] = j #cj: to my understanding, this is the same as np.argsort(x) and then reverse it
            lambdaPerArea_temp[j] = -1

        lambdaPerAreaHigh = lambdaPerArea[locationWithHigestLambdaPerArea[0]]
        distance_tau = self.walkingSpeed/3600 * (self.threshold - self.volResDelay) # in kms
        totalToAdd = 0.0

        remainingNu = 1.0

        for i in range(len(lambdaPerArea)):
            lambdaPerAreaLow = lambdaPerArea[locationWithHigestLambdaPerArea[i]]
            intersection = math.log(lambdaPerAreaHigh/lambdaPerAreaLow)/(self.nVolunteers*math.pi*distance_tau*distance_tau/self.area[locationWithHigestLambdaPerArea[0]])

            if (i == len(lambdaPerArea)-1):
                nextIntersection = 1.0
            else:
                nextIntersection = math.log(lambdaPerAreaHigh/lambdaPerArea[locationWithHigestLambdaPerArea[i+1]])/(self.nVolunteers*math.pi*distance_tau*distance_tau/self.area[locationWithHigestLambdaPerArea[0]])
            totalToAdd += self.area[locationWithHigestLambdaPerArea[i]]
            requiredToAdd = totalToAdd*(nextIntersection-intersection)

            if (requiredToAdd < remainingNu):
                for j in range(i+1):
                    nu[locationWithHigestLambdaPerArea[j]] += (nextIntersection-intersection)*self.area[locationWithHigestLambdaPerArea[j]]
                    remainingNu -= (nextIntersection-intersection)*self.area[locationWithHigestLambdaPerArea[j]]
            else:
                ratio = (remainingNu/requiredToAdd)
                for j in range(i+1):
                    nu[locationWithHigestLambdaPerArea[j]] += (nextIntersection-intersection)*ratio*self.area[locationWithHigestLambdaPerArea[j]]
                    remainingNu -= (nextIntersection-intersection)*ratio*self.area[locationWithHigestLambdaPerArea[j]]
                break
        return nu

    def Greedy_lateArrival(self) -> list: #CAREFUL - DOES NOT USE ALPHA
        """
        Compute volunteer distrubution by optimizing late arrival rate
        for each location using greedy algorithm with discretized stepsizes.

        Returns
        -------
        nu : list
            Optimized volunteer distribution.
        """
        nu = [0.0] * self.nLocations
        additional_onTimeArrival_temp = [0.0] * self.nLocations
        additional_onTimeArrival = np.asarray(additional_onTimeArrival_temp)

        delta = 1 / self.nSteps
        for i in range(self.nLocations):
            additional_onTimeArrival[i] = self.AdditionalOnTimeArrivalInLocation(nu[i],i, delta)

        locationToAddVolunteers = additional_onTimeArrival.argmax()

        found = False

        for z in range(0,self.nSteps):
            locationToAddVolunteers = additional_onTimeArrival.argmax()

            if (found == False and locationToAddVolunteers == 0):
                print('first time location 0 is selected: ', z)
                found = True
            #print(locationToAddVolunteers)
            nu[locationToAddVolunteers] += delta
            additional_onTimeArrival[locationToAddVolunteers] = self.AdditionalOnTimeArrivalInLocation(nu[locationToAddVolunteers],locationToAddVolunteers, delta)
            if (z % 1000 == 0):
                print(z, locationToAddVolunteers)
        return nu

    def Greedy_lateArrival_N(self, nVol: int) -> list:
        """
        Compute volunteer distrubution with specified number of volunteers by
        optimizing late arrival rate for each location using greedy algorithm.

        Parameters
        ----------
        nVol : int
            Number of volunteers.

        Returns
        -------
        nu : list
            Optimized volunteer distribution.
        """
        #This function does temporarily change the number of volunteers in this class
        originalNVol = self.nVolunteers
        self.nVolunteers = nVol
        nu = self.Greedy_lateArrival()
        self.nVolunteers = originalNVol
        return nu

    def evaluateLateArrival_as_in_java(self, nu: list) -> float:
        """
        Evaluate sum of probability that a response time is greater than $\tau$

        Parameters
        ----------
        nu : list
            Volunteer location distribution.

        Returns
        -------
        result : float
            Sum of probability that a response time is greater than $\tau$.

        """

        result = 0.0

        for i in range(self.nLocations):
            result += self.LateArrivalInLocation(nu[i], i)

        return result

    #----------------------------------------------------------------
    #       First responder
    #----------------------------------------------------------------

    def ProbNoResponseInLocation(self, nu: float, loc: int) -> float:
        """
        Compute probability of no response for a location.

        Parameters
        ----------
        nu : float
            Probability that volunteers appear in given location.
        loc : int
            Index of location.

        Returns
        -------
        float
            Probability that there is no ambulance or volunteer response.
        """
        pNoAmbu = (self.busyProb**self.nAmbulances)
        temp = - self.nVolunteers * math.pi * self.maxDispatchDistance * self.maxDispatchDistance / self.area[loc]
        pNoVolunteer = math.exp(temp*nu)

        return pNoAmbu * pNoVolunteer


    def AmbuFirstInLocation(self, nu: float, loc: int) -> float:
        """
        Compute probability that ambulance shows up first for a location.

        Parameters
        ----------
        nu : float
            Probability that volunteers appear in given location.
        loc : int
            Index of location.

        Returns
        -------
        float
            Probability that ambulance reaches patient before volunteer.
        """
        result = 0.0

        respTimesAmbulances = [0] * self.nAmbulances
        nAmbuFound = 0
        for i in range(0,self.nBases):
            if (self.ambus[i] > 0) :
                count = self.ambus[i]
                while (count>0):
                    respTimesAmbulances[nAmbuFound] = self.travelTime_A[i][loc] + self.ambuResDelay

                    count = count - 1
                    nAmbuFound = nAmbuFound + 1

        if (nAmbuFound != self.nAmbulances):
            print('Not all ambulances found in the array ambus')

        sortedRespTime = np.array(respTimesAmbulances)
        sortedRespTime.sort()

        for i in range(0, self.nAmbulances):
            #i closest ambulances are busy, i+1-th ambulance is first to be available

            if (i < self.nAmbulances):
                prob = (1-self.busyProb) * (self.busyProb**i)
                responsetime_closest_available_ambu = sortedRespTime[i]
            else:
                prob = self.busyProb**i
                responsetime_closest_available_ambu = float('inf')

            distance_t = self.walkingSpeed/3600 * (responsetime_closest_available_ambu - self.volResDelay) # in kms
            if (distance_t < 0):
                P_volunteer_responds_within_t = 0.0
            elif (distance_t > self.maxDispatchDistance):
                temp = - self.nVolunteers * math.pi * self.maxDispatchDistance * self.maxDispatchDistance / self.area[loc]
                P_volunteer_responds_within_t = 1 - math.exp(temp*nu)
            else:
                temp = - self.nVolunteers * math.pi * distance_t * distance_t / self.area[loc]
                P_volunteer_responds_within_t = 1 - math.exp(temp*nu)

            result += (1-P_volunteer_responds_within_t)*prob

        return result

    def VolunteerFirstInLocation(self, nu: float, loc: int) -> float:
        """
        Compute probability that volunteer shows up first for a location.

        Parameters
        ----------
        nu : float
            Probability that volunteers appear in given location.
        loc : int
            Index of location.

        Returns
        -------
        float
            Probability that volunteer reaches patient before ambulance.
        """
        result = 0.0

        respTimesAmbulances = [0] * self.nAmbulances
        nAmbuFound = 0
        for i in range(0,self.nBases):
            if (self.ambus[i] > 0) :
                count = self.ambus[i]
                while (count>0):
                    respTimesAmbulances[nAmbuFound] = self.travelTime_A[i][loc] + self.ambuResDelay
                    count = count - 1
                    nAmbuFound = nAmbuFound + 1

        if (nAmbuFound != self.nAmbulances):
            print('Not all ambulances found in the array ambus')

        sortedRespTime = np.array(respTimesAmbulances)
        sortedRespTime.sort()

        for i in range(0, self.nAmbulances+1):
            #i closest ambulances are busy, i+1-th ambulance is first to be available

            if (i < self.nAmbulances):
                prob = (1-self.busyProb) * (self.busyProb**i)
                responsetime_closest_available_ambu = sortedRespTime[i]
            else:
                prob = self.busyProb**i
                responsetime_closest_available_ambu = float('inf')

            distance_t = self.walkingSpeed/3600 * (responsetime_closest_available_ambu - self.volResDelay) # in kms
            if (distance_t < 0):
                P_volunteer_responds_within_t = 0.0
            elif (distance_t > self.maxDispatchDistance):
                temp = - self.nVolunteers * math.pi * self.maxDispatchDistance * self.maxDispatchDistance / self.area[loc]
                P_volunteer_responds_within_t = 1 - math.exp(temp*nu)
            else:
                temp = - self.nVolunteers * math.pi * distance_t * distance_t / self.area[loc]
                P_volunteer_responds_within_t = 1 - math.exp(temp*nu)

            result += P_volunteer_responds_within_t*prob

        return result

    def evaluateProbVolFirst(self, nu: list) -> float:
        """
        Compute probability that volunteer shows up first for the whole city.

        Parameters
        ----------
        nu : list
            Volunteer location distribution.

        Returns
        -------
        float
            Probability that volunteer reaches patient before ambulance.
        """
        result = 0.0

        for i in range(self.nLocations):
            result += self.VolunteerFirstInLocation(nu[i], i)*self.lambdaL[i]

        return result

    def evaluateProbAmbuFirst(self, nu: list) -> float:
        """
        Compute probability that ambulance shows up first for the whole city.

        Parameters
        ----------
        nu : list
            Volunteer location distribution.

        Returns
        -------
        float
            Probability that ambulance reaches patient before volunteers.
        """
        result = 0.0

        for i in range(self.nLocations):
            result += self.AmbuFirstInLocation(nu[i], i)*self.lambdaL[i]

        return result

    def evaluateProbNoResponse(self, nu: list) -> float:
        """
        Compute probability that there is no response for the whole city.

        Parameters
        ----------
        nu : list
            Volunteer location distribution.

        Returns
        -------
        float
            Probability that there is no volunteer or ambulance response.
        """
        result = 0.0

        for i in range(self.nLocations):
            result += self.ProbNoResponseInLocation(nu[i], i)*self.lambdaL[i]

        return result

    #----------------------------------------------------------------
    #       Greedy Survivability
    #----------------------------------------------------------------

    #-----------------
    #   deMaio
    #-----------------

    def DeMaioSurvivalProb(self, responseTimeInSeconds: int) -> float:
        """
        Calculate probibility of survival (deMaio et al. (2003)).

        Parameters
        ----------
        responseTimeInSeconds : int
            The time between patient collapse and ambulance arrival
            measured in seconds.

        Returns
        -------
        survivalProb : float
            Probibility of survival.
        """
        timeAfterCallInitiation = responseTimeInSeconds - 60
        survivalProb = 1 / (1 + np.exp(0.679 + 0.262 * timeAfterCallInitiation / 60))
        return survivalProb

    '''
    always compute deMaio WITHOUT ambus
    '''
    def SurvivalInLocation_deMaio_alternative(self, nu: float, loc: int) -> float:
        """
        Compute survival probability for a location.

        Parameters
        ----------
        nu : float
            Probability that volunteers appear in given location.
        loc : int
            Index of location.

        Returns
        -------
        float
            Probability of survival.
        """
        result = 0.0
        """
        Ambulance Contribution
        """

        ## responsetimeForEachBase
        responsetimeForEachBase = np.array(self.travelTime_A)[:self.nBases, loc] + self.ambuResDelay

        ## ambuCount in each time slot
        ambuCountForTime = np.zeros(max(responsetimeForEachBase)+1)
        for i, responsetime in enumerate(responsetimeForEachBase):
            ambuCountForTime[responsetime] += self.ambus[i]

        # probAmbu in each time slot
        probAmbu = 1 - self.busyProb ** ambuCountForTime

        ## distance in each time slot
        distance = self.walkingSpeed / 3600 * (np.arange(max(responsetimeForEachBase)+1) - self.volResDelay)
        np.clip(distance, 0, self.maxDispatchDistance, out=distance)

        # probNoVolBefore in each time slot
        probNoVolBefore = np.exp(-self.nVolunteers * math.pi * distance ** 2 / self.area[loc] * nu)

        # deMaioSurvival in each time slot
        survivalProb = self.DeMaioSurvivalProb(np.arange(max(responsetimeForEachBase)+1))

        # probNoAmbuBefore
        ## accumAmbuCount
        cumsumAmbuCount = np.cumsum(ambuCountForTime)
        cumsumAmbuCount = np.insert(cumsumAmbuCount, 0, 0)[:-1] # add 0 first and cut last
        probNoAmbuBefore = self.busyProb ** cumsumAmbuCount

        # calculate result
        result += sum(probNoAmbuBefore * probAmbu * probNoVolBefore * survivalProb)

        """
        Volunteer Contribution
        """

        # P_closest_between_t_and_tmin1
        timeslot = np.arange(1, self.maxDispatchTime+1)

        distance_t = self.walkingSpeed / 3600 * (timeslot - self.volResDelay)
        np.clip(distance_t, 0, self.maxDispatchDistance, out=distance_t)
        tempT =  - self.nVolunteers * math.pi * distance_t * distance_t / self.area[loc]

        distance_tmin1 = self.walkingSpeed / 3600 * ((timeslot - 1) - self.volResDelay)
        np.clip(distance_tmin1, 0, self.maxDispatchDistance, out=distance_tmin1)
        tempTmin1 = - self.nVolunteers * math.pi * distance_tmin1 * distance_tmin1 / self.area[loc]

        P_closest_between_t_and_tmin1 = np.exp(tempTmin1*nu) - np.exp(tempT*nu)

        # deMaioSurvivalProb
        survivalProb = self.DeMaioSurvivalProb(timeslot - 1)

        # Calculate result
        result += sum(probNoAmbuBefore[1:self.maxDispatchTime+1] * P_closest_between_t_and_tmin1 * survivalProb)

        return self.lambdaL[loc] * result

    def Greedy_deMaio(self) -> list: #CAREFUL - DOES NOT USE ALPHA
        """
        Compute volunteer distrubution by optimizing deMaio survival rate
        for each location using greedy algorithm.

        Returns
        -------
        nu : list
             A list of optimized volunteer distrubution.
        """

        nu = [0.0] * self.nLocations
        additional_survival_temp = [0.0] * self.nLocations
        survival = [0.0] * self.nLocations
        additional_survival = np.asarray(additional_survival_temp)

        delta = 1 / self.nSteps
        for i in range(self.nLocations):
            survival[i] = self.SurvivalInLocation_deMaio_alternative(nu[i], i)
            additional_survival[i] = self.SurvivalInLocation_deMaio_alternative(nu[i]+delta, i) - survival[i]

            #additional_survival[i] = self.AdditionalSurvivalInLocation_deMaio(nu[i],i, delta)

        locationToAddVolunteers = additional_survival.argmax()

        found = False

        for z in range(0,self.nSteps):
            locationToAddVolunteers = additional_survival.argmax()
            if (found == False and locationToAddVolunteers == 0):
                print('first time location 0 is selected: ', z)
                found = True
            #print(locationToAddVolunteers)
            nu[locationToAddVolunteers] += delta
            #additional_survival[locationToAddVolunteers] = self.AdditionalSurvivalInLocation_deMaio(nu[locationToAddVolunteers],locationToAddVolunteers, delta)
            survival[locationToAddVolunteers] = survival[locationToAddVolunteers] + additional_survival[locationToAddVolunteers]
            additional_survival[locationToAddVolunteers] = self.SurvivalInLocation_deMaio_alternative(nu[locationToAddVolunteers]+delta, locationToAddVolunteers) - survival[locationToAddVolunteers]
            if (z % 1000 == 0):
                print(z, locationToAddVolunteers)
        return nu

    def Greedy_deMaio_N(self, nVol: int) -> list:
        """
        Compute optimal volunteer distribution with different number
        of volunteers.

        Parameters
        ----------
        nvol : int
            Alternative number of volunteers.

        Returns
        -------
        nu : list
            A list of optimized volunteer distrubution.
        """

        #This function does temporarily change the number of volunteers in this class
        originalNVol = self.nVolunteers
        self.nVolunteers = nVol
        nu = self.Greedy_deMaio()
        self.nVolunteers = originalNVol
        return nu

    def evaluatedeMaio_as_in_java(self, nu: list) -> float:
        """
        Evaluate sum of deMaio survival probability.

        Parameters
        ----------
        nu : list
            Volunteer location distribution.

        Returns
        -------
        float
            Sum of survival rate.
        """

        result = 0.0

        for i in range(self.nLocations):
            #result += self.SurvivalInLocation_deMaio(nu[i], i)
            result += self.SurvivalInLocation_deMaio_alternative(nu[i], i)

        return result



    #-----------------
    #   Waalewijn model 1 - situation with one type of patient, two types of responses: CPR (volunteer or ambu) and EMS (ambu only)
    #-----------------

    # easier to work on the assumption this is False
    # TODO: Add code when this is true


    WaalewijnNonzeroSurvivalWhenAllAmbusBusy = False

    #so-called Waalewijn Model 1, I've changd the input to reflect volunteer and ems
    def Waalewijn_model1(self, tVOL_in_sec: int, tEMS_in_sec: int) -> float:
        """
        Calculate probibility of survival (Waalewijn et al. (2001)).

        Parameters
        ----------
        responseTimeInSeconds : int
            The time between patient collapse and ambulance arrival
            measured in seconds.

        Returns
        -------
        survivalProb : float
            Probibility of survival.
        """

        # who-ever gets there first
        tCPR_in_sec = np.array(np.minimum(tVOL_in_sec, tEMS_in_sec))

        tCPR_in_sec[tCPR_in_sec == float('inf')] = 0

        tCPR = tCPR_in_sec / 60.0
        tEMS = tEMS_in_sec / 60.0


        #tCPR = self.volResDelay / 60.0

        survivalProb = 1 / (1 + np.exp(0.04 + 0.3 * tCPR + 0.14 * (tEMS - tCPR)))

        return survivalProb

    def Greedy_Waalewijn_model1(self) -> list:
        """
        Compute volunteer distrubution by optimizing Waalewijn survival rate
        for each location using greedy algorithm.

        Returns
        -------
        nu : list
             A list of optimized volunteer distrubution.
        """
        if (sum(self.ambus) == 0):
            print("Warning: Waalewijn model 1 is not appropriate when there are no ambulances." ,
                "Survival will be 0 regardless of where the volunteers are.")
            print("I will return an empty volunteer distribution")

            return [0.0] * self.nLocations

        nu = np.zeros((self.nLocations))
        survival = np.zeros((self.nLocations))
        additional_survival_temp = np.zeros((self.nLocations))
        additional_survival = np.asarray(additional_survival_temp)

        delta = 1 / self.nSteps
        for i in range(self.nLocations):
            survival[i] = self.SurvivalInLocation_Waalewijn_model1_alternative(nu[i], i)
            additional_survival[i] = self.SurvivalInLocation_Waalewijn_model1_alternative(nu[i]+delta, i) - survival[i]
            # additional_survival[i] = self.AdditionalSurvivalInLocation_Waalewijn_model1(nu[i],i, delta)

        locationToAddVolunteers = additional_survival.argmax()

        print(self.nSteps)

        locationRespTimes = {}

        start = time.time()

        for z in range(0,self.nSteps):
            #print("Step", z)
            locationToAddVolunteers = additional_survival.argmax()
            #print(locationToAddVolunteers)
            nu[locationToAddVolunteers] += delta
            survival[locationToAddVolunteers] += additional_survival[locationToAddVolunteers]
            additional_survival[
                locationToAddVolunteers
            ] = self.SurvivalInLocation_Waalewijn_model1_alternative(
                nu[locationToAddVolunteers]+delta, locationToAddVolunteers
            ) - survival[locationToAddVolunteers]
            #additional_survival[locationToAddVolunteers] = self.AdditionalSurvivalInLocation_Waalewijn_model1(nu[locationToAddVolunteers],locationToAddVolunteers,delta)
            if (z % 1000 == 0):
                print(z, locationToAddVolunteers)

        end = time.time() - start
        time_per_z = end / self.nSteps
        print(time_per_z)
        return nu

    def Greedy_Waalewijn_model1_N(self, nVol: int) -> list:
        """
        Compute optimal volunteer distribution with different number
        of volunteers.

        Parameters
        ----------
        nvol : int
            Alternative number of volunteers.

        Returns
        -------
        nu : list
            A list of optimized volunteer distrubution.
        """
        #This function does temporarily change the number of volunteers in this class
        if (nVol == 0): #save time:
            return [0.0] * self.nLocations

        originalNVol = self.nVolunteers
        self.nVolunteers = nVol
        nu = self.Greedy_Waalewijn_model1()
        self.nVolunteers = originalNVol
        return nu


    def calcWaalewijnSurvivalMatrix(self):
        """

        """

        maxDispatchTime = self.maxDispatchTime
        max_responsetimeForEachBase = 3000

        survivalMatrix = np.zeros((maxDispatchTime+1, max_responsetimeForEachBase+1))
        for t in range(1, maxDispatchTime+1):
            for resp in range(1, max_responsetimeForEachBase+1):
                if resp < t:
                    survivalMatrix[t, resp] = self.Waalewijn_model1(resp, resp)
                else:
                    survivalMatrix[t, resp] = self.Waalewijn_model1(t-1, resp)
        survivalMatrix = survivalMatrix[1:,:]

        return survivalMatrix


    def SurvivalInLocation_Waalewijn_model1_alternative(self, nu: float, loc: int) -> float:
        """
        Compute Waalewijn survival probability for a location.

        Parameters
        ----------
        nu : float
            Probability that volunteers appear in given location.
        loc : int
            Index of location.

        Returns
        -------
        float
            Probability of survival.
        """

        """
        Closest volunteers contribution
        """
        ## responsetimeForEachBase
        responsetimeForEachBase = np.array(self.travelTime_A)[:self.nBases, loc] + self.ambuResDelay
        ambuCountForTime = np.zeros(max(responsetimeForEachBase)+1)
        for i, responsetime in enumerate(responsetimeForEachBase):
            ambuCountForTime[responsetime] += self.ambus[i]

        maxTime = self.maxDispatchTime

        ## P_closest_between_t_and_tmin1
        timeslot = np.arange(1, maxTime+1)
        ### tempT
        distance_t = self.walkingSpeed / 3600 * (timeslot - self.volResDelay)
        np.clip(distance_t, 0, self.maxDispatchDistance, out=distance_t)
        tempT = - self.nVolunteers * math.pi * distance_t * distance_t / self.area[loc]
        ### tempTmin1
        distance_tmin1 = self.walkingSpeed / 3600 * ((timeslot - 1) - self.volResDelay)
        np.clip(distance_tmin1, 0, self.maxDispatchDistance, out=distance_tmin1)
        tempTmin1 = - self.nVolunteers * math.pi * distance_tmin1 * distance_tmin1 / self.area[loc]
        ### P_closest_between_t_and_tmin1
        P_closest_between_t_and_tmin1 = np.exp(tempTmin1*nu) - np.exp(tempT*nu)

        ## probClosestAmbuAtThisBase
        ### probNoAmbuBefore
        cumsumAmbuCount = np.cumsum(ambuCountForTime)
        cumsumAmbuCount = np.insert(cumsumAmbuCount, 0, 0)[:-1] # add 0 first and cut last
        probNoAmbuBefore = self.busyProb ** cumsumAmbuCount
        ### probAmbu
        probAmbu = 1 - self.busyProb ** ambuCountForTime
        ### probClosestAmbuAtThisBase
        probClosestAmbuAtThisBase = probNoAmbuBefore * probAmbu

        ## self.Waalewijn_model1
        survivalMatrix = self.waalewijnSurvivalMatrix[:, :max(responsetimeForEachBase)+1]

        P_closest_between_t_and_tmin1_extend = np.tile(P_closest_between_t_and_tmin1.T, (max(responsetimeForEachBase)+1, 1)).T
        probClosestAmbuAtThisBase_extend = np.tile(probClosestAmbuAtThisBase, (maxTime, 1))
        result = np.sum(P_closest_between_t_and_tmin1_extend * probClosestAmbuAtThisBase_extend * survivalMatrix)

        """
        This is to account for situation without any volunteers
        """
        ## probNoVolBefore in each time slot
        probNoVolBefore = np.exp(-self.nVolunteers * math.pi * self.maxDispatchDistance ** 2 / self.area[loc] * nu)

        ## probClosestAmbu
        ### probAmbu in each time slot
        probAmbu = 1 - self.busyProb ** ambuCountForTime
        ### probNoAmbuBefore
        probNoAmbuBefore = self.busyProb ** cumsumAmbuCount
        ### probClosestAmbu
        probClosestAmbu = probAmbu * probNoAmbuBefore

        ## survivalProb in each time slot
        timeslot = np.arange(max(responsetimeForEachBase)+1)
        survivalProb = self.Waalewijn_model1(timeslot, timeslot)

        ## calculate result
        result += sum(probNoVolBefore * probClosestAmbu * survivalProb)

        return self.lambdaL[loc] * result


    def evaluateWaalewijn_model1_as_in_java(self, nu: list) -> float:
        """
        Evaluate sum of Waalewijn survival probability.

        Parameters
        ----------
        nu : list
            Volunteer location distribution.

        Returns
        -------
        float
            Sum of survival rate.
        """
        if (sum(self.ambus) == 0):
            print("Warning: You are running without ambulances, which means Waalewijn model 1 is not appropriate." ,
                "Survival will be 0 regardless of where the volunteers are.")

        result = 0.0
        for i in range(self.nLocations):
            result += self.SurvivalInLocation_Waalewijn_model1_alternative(nu[i], i)

        return result

    def Survival_Waalewijn_Upper(self) -> float:
        """
        Compute upper limit of Waalewijn survival probability with the given
        ambulance distribution and assume no volunteer travel time.

        Returns
        -------
        float
            Upper limit of Waalewijn survival.
        """

        """
        Closest volunteers contribution
        """
        result = 0
        for loc in range(self.nLocations):
            ## responsetimeForEachBase
            responsetimeForEachBase = np.array(self.travelTime_A)[:self.nBases, loc] + self.ambuResDelay
            ambuCountForTime = np.zeros(max(responsetimeForEachBase)+1)
            for i, responsetime in enumerate(responsetimeForEachBase):
                ambuCountForTime[responsetime] += self.ambus[i]

            ## probClosestAmbuAtThisBase
            ### probNoAmbuBefore
            cumsumAmbuCount = np.cumsum(ambuCountForTime)
            cumsumAmbuCount = np.insert(cumsumAmbuCount, 0, 0)[:-1] # add 0 first and cut last

            ## probClosestAmbu
            ### probAmbu in each time slot
            probAmbu = 1 - self.busyProb ** ambuCountForTime
            ### probNoAmbuBefore
            probNoAmbuBefore = self.busyProb ** cumsumAmbuCount
            ### probClosestAmbu
            probClosestAmbu = probAmbu * probNoAmbuBefore

            ## survivalProb in each time slot
            timeslot = np.arange(max(responsetimeForEachBase)+1)
            time_CPR = np.minimum(self.volResDelay, timeslot)
            survivalProb = self.Waalewijn_model1(time_CPR, timeslot)

            ## calculate result
            result += self.lambdaL[loc] * sum(probClosestAmbu * survivalProb)

        return result

    #-----------------
    #   Waalewijn model 1 - Fix Ambulance response time, 780s
    #-----------------

    def SurvivalInLocation_Waalewijn_FixEMS(self, nu: float, loc: int) -> float:
        """
        Compute Waalewijn survival probability with fixed 13 minutes ambulance
        response time for a location.

        Parameters
        ----------
        nu : float
            Probability that volunteers appear in given location.
        loc : int
            Index of location.

        Returns
        -------
        float
            Probability of Waalewijn survival with fixed ambulance response.
        """
        result = 0.0

        """
        Volunteer Contribution
        """

        # P_closest_between_t_and_tmin1
        timeslot = np.arange(1, self.maxDispatchTime+1)

        distance_t = self.walkingSpeed / 3600 * (timeslot - self.volResDelay)
        np.clip(distance_t, 0, self.maxDispatchDistance, out=distance_t)
        tempT =  - self.nVolunteers * math.pi * distance_t * distance_t / self.area[loc]

        distance_tmin1 = self.walkingSpeed / 3600 * ((timeslot - 1) - self.volResDelay)
        np.clip(distance_tmin1, 0, self.maxDispatchDistance, out=distance_tmin1)
        tempTmin1 = - self.nVolunteers * math.pi * distance_tmin1 * distance_tmin1 / self.area[loc]

        P_closest_between_t_and_tmin1 = np.exp(tempTmin1*nu) - np.exp(tempT*nu)

        # deMaioSurvivalProb
        survivalProb = self.Waalewijn_model1(timeslot - 1, 780)

        # Calculate result
        result += sum(P_closest_between_t_and_tmin1 * survivalProb)

        # Probability that there is no volunteers
        probNoVolunteer = np.exp(-self.nVolunteers * math.pi * distance_t[-1] ** 2 / self.area[loc] * nu)
        result += probNoVolunteer * self.Waalewijn_model1(780, 780)

        return self.lambdaL[loc] * result

    def Greedy_Waalewijn_FixEMS(self) -> list: #CAREFUL - DOES NOT USE ALPHA
        """
        Compute volunteer distrubution by optimizing Waalwijn survival rate
        with fixed 13 min ambulance response for each location using greedy
        algorithm.

        Returns
        -------
        nu : list
             A list of optimized volunteer distrubution.
        """

        nu = [0.0] * self.nLocations
        additional_survival_temp = [0.0] * self.nLocations
        survival = [0.0] * self.nLocations
        additional_survival = np.asarray(additional_survival_temp)

        delta = 1 / self.nSteps
        for i in range(self.nLocations):
            survival[i] = self.SurvivalInLocation_Waalewijn_FixEMS(nu[i], i)
            additional_survival[i] = self.SurvivalInLocation_Waalewijn_FixEMS(nu[i]+delta, i) - survival[i]

            #additional_survival[i] = self.AdditionalSurvivalInLocation_deMaio(nu[i],i, delta)

        locationToAddVolunteers = additional_survival.argmax()

        found = False

        for z in range(0,self.nSteps):
            locationToAddVolunteers = additional_survival.argmax()
            if (found == False and locationToAddVolunteers == 0):
                print('first time location 0 is selected: ', z)
                found = True
            #print(locationToAddVolunteers)
            nu[locationToAddVolunteers] += delta
            #additional_survival[locationToAddVolunteers] = self.AdditionalSurvivalInLocation_deMaio(nu[locationToAddVolunteers],locationToAddVolunteers, delta)
            survival[locationToAddVolunteers] = survival[locationToAddVolunteers] + additional_survival[locationToAddVolunteers]
            additional_survival[locationToAddVolunteers] = self.SurvivalInLocation_Waalewijn_FixEMS(nu[locationToAddVolunteers]+delta, locationToAddVolunteers) - survival[locationToAddVolunteers]
            if (z % 1000 == 0):
                print(z, locationToAddVolunteers)
        return nu

    def Greedy_Waalewijn_FixEMS_N(self, nVol: int) -> list:
        """
        Compute optimal volunteer distribution for Waalewijn with fixed EMS
        response with different number of volunteers.

        Parameters
        ----------
        nvol : int
            Alternative number of volunteers.

        Returns
        -------
        nu : list
            A list of optimized volunteer distrubution.
        """

        #This function does temporarily change the number of volunteers in this class
        originalNVol = self.nVolunteers
        self.nVolunteers = nVol
        nu = self.Greedy_Waalewijn_FixEMS()
        self.nVolunteers = originalNVol
        return nu

    def evaluateWaalewijn_FixEMS_as_in_java(self, nu: list) -> float:
        """
        Evaluate sum of Waalwijn survival probability with fixed EMS response
        time (13 minutes).

        Parameters
        ----------
        nu : list
            Volunteer location distribution.

        Returns
        -------
        float
            Sum of Waalewijn survival rate.
        """

        result = 0.0

        for i in range(self.nLocations):
            result += self.SurvivalInLocation_Waalewijn_FixEMS(nu[i], i)

        return result
