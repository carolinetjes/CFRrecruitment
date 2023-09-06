"""
optimization_profile_time.py
=========================================

Module that used to compute ideal volunteer recruitment distribution when we
consider profile with different time segments.
"""


import math
import numpy as np
from .instance import Instance
from gurobipy import *


class ProfileOptTime:
    """
    Computes the optimal volunteer recruitment distribution with given
    profile under both coverage and survival probability objectives.

    Args
    ----
    inst : Instance
        An Instance object that contains information read from the Input files.
    numTimeSeg : int
        Number of time segments we are considering.
    alphaL : list
        A list of volunteer acceptance probability during different time
        segments.
    lambdaL : list
        A list of incident distribution during different time segments.
    OHCAProbL : list
        A list of probability that OHCA occurs during each time segments.
    profileL : list
        List of profiles that indicates where volunteers stays during
        different time segments.

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
    numTimeSeg : int
        Number of time segments we are considering.
    alphaL : list
        A list of volunteer acceptance probability during different time
        segments.
    lambdaL : list
        A list of incident distribution during different time segments.
    OHCAProbL : list
        A list of probability that OHCA occurs during each time segments.
    profileL : list
        List of profiles that indicates where volunteers stays during
        different time segments.
    nAmbulances : int
        Number of ambulances.
    maxDispatchTime : int
        Max dispatch radius in time by assuming constant walking speed for
        volunteers.
    A : list
        List of tranpose of profiles.
    """
    def __init__(self, inst: Instance, numTimeSeg: int, alphaL: list,
                 lambdaL : list, OHCAProbL: list, profileL: list) -> None:

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

        # Set parameters
        self.numTimeSeg = numTimeSeg
        self.alphaL = alphaL # maybe need a clearer name
        self.lambdaL = lambdaL
        self.OHCAProbL = OHCAProbL
        self.profileL = profileL

        # Additional parameters
        self.nAmbulances = sum(self.ambus)
        self.maxDispatchTime = int((self.maxDispatchDistance/self.walkingSpeed)*3600 + self.volResDelay)
        self.A = [self.profileL[i].T for i in range(self.numTimeSeg)]


    #----------------------------------------------------------------
    #       Helper functions
    #----------------------------------------------------------------

    def DeMaioSurvivalProb(self, responseTimeInSeconds):
        """
        Calculate probibility of survival (deMaio et al. (2003)).

        Parameters
        ----------
        responseTimeInSeconds : int
            Time from patient collapse to EMS arrival.

        Returns
        -------
        survival : float
            Survival probability of OHCA patient.
        """
        timeAfterCallInitiation = responseTimeInSeconds - 60
        survival = 1 / (1 + np.exp(0.679 + 0.262 * timeAfterCallInitiation / 60))
        return survival

    def Waalewijn_model1(self, tVOL_in_sec, tEMS_in_sec):
        """
        Calculate probibility of survival (Waalewijn et al. (2001)).

        Parameters
        ----------
        tVOL_in_sec : int
            Time from patient collapse to Volunteer arrival.
        tEMS_in_sec : int
            Time from patient collapse to EMS arrival.

        Returns
        -------
        survival : float
            Survival probability of OHCA patient.
        """
        # who-ever gets there first
        tCPR_in_sec = np.array(np.minimum(tVOL_in_sec, tEMS_in_sec))

        tCPR_in_sec[tCPR_in_sec == float('inf')] = 0

        tCPR = tCPR_in_sec / 60.0
        tEMS = tEMS_in_sec / 60.0

        survivalProb = 1 / (1 + np.exp(0.04 + 0.3 * tCPR + 0.14 *
                                       (tEMS - tCPR)))

        return survivalProb

    def standard(self, size: int, index: int) -> np.array:
        """
        Generate standard unit vector with specified size and index.

        Parameters
        ----------
        size : int
            Size of the output array.
        index : int
            Index that needs to have value 1.

        Returns
        -------
        x : numpy.array
            A standard unit vector with specific index being 1.
        """

        x = np.zeros(size)
        x[index] = 1.0
        return x

    def getLowerbound(self, D):
        """
        Computes a lower bound on the objective value given the cuts generated
        by all the iterates up to date.

        Parameters
        ----------
        D : list
            An array that stores all the cuts

        Returns
        -------
        float
            Lower bound on the objective values given all the cuts up to date.
        """

        m = Model("lower_bound")
        m.Params.LogToConsole = 0
        z = m.addVar(lb=-GRB.INFINITY, vtype = GRB.CONTINUOUS, name='z')
        x = [0] * self.nLocations
        for i in range(self.nLocations):
            x[i] = m.addVar(lb = 0.0, vtype=GRB.CONTINUOUS, name='x'+str(i))
        m.update()

        Obj = z
        m.setObjective(Obj, GRB.MINIMIZE)

        m.addConstr(sum(x) == 1, name = 'probDist')
        for g, y, val in D:
            expr = LinExpr()
            expr += val
            for i in range(self.nLocations):
                expr += g[i] * (x[i] - y[i])

            m.addConstr(z >= expr)
        m.optimize()
        return m.objVal


    #----------------------------------------------------------------
    #       Maximize DeMaio (separable, continuous evaluation)
    #----------------------------------------------------------------

    def pDeMaio(self, x):
        """
        Compute the negative of DeMaio survival we want to minimize.

        Parameters
        ----------
        x : numpy.array
            An array that represents the volunteer **recruitment** distribution.

        Returns
        -------
        float
            Negative of DeMaio survival probability with given recruitment
            distribution.
        """

        nu_L = [self.A[i] @ x for i in range(self.numTimeSeg)]
        return self.contDeMiao(nu_L)

    def gradientPDeMaio(self, x):
        """
        Compute the gradient of the DeMaio objective function.

        Parameters
        ----------
        x : numpy.array
            An array that represents the volunteer **recruitment** distribution.

        Returns
        -------
        numpy.array
            Gradient of DeMaio survival objective as a function of x.
        """

        nu_L = [self.A[i] @ x for i in range(self.numTimeSeg)]
        g = self.gradientContDeMaio(nu_L)
        result = [self.A[i].T @ g[i] for i in range(self.numTimeSeg)]
        return sum(result)

    def evaluatePDeMaio(self, x):
        """
        Compute the DeMaio survival probability.

        Parameters
        ----------
        x : numpy.array
            An array that represents the volunteer **recruitment** distribution.

        Returns
        -------
        float
            Survival probability with given recruitment disbution.
        """
        return -self.pDeMaio(x)


    def contDeMiao(self, nu_L) -> float:
        """
        Computes the DeMaio survival objective value we want to minimize.

        Parameters
        ----------
        nu_L : list
            A list of volunteer **location** distribution during different
            time segment.

        Returns
        -------
        float
            Negative of DeMaio survival probability.
        """
        result = 0

        for T in range(self.numTimeSeg):
            for loc in range(self.nLocations):

                for t in range(1, int(self.maxDispatchTime) + 1):
                    #i closest ambulances are busy, i+1-th ambulance is first to be available

                    distance_t = self.walkingSpeed / 3600 * (t - self.volResDelay)  # in kms
                    distance_tmin1 = self.walkingSpeed / 3600 * (t - 1 - self.volResDelay)  # in kms
                    if (distance_t <= 0):
                        distance_t = 0
                    elif (distance_t > self.maxDispatchDistance):
                        distance_t = self.maxDispatchDistance
                    if (distance_tmin1 <= 0):
                        distance_tmin1 = 0
                    elif (distance_tmin1 > self.maxDispatchDistance):
                        distance_tmin1 = self.maxDispatchDistance

                    tempT = -self.nVolunteers * self.alphaL[T] * math.pi * distance_t * distance_t / self.area[loc]
                    tempTmin1 = -self.nVolunteers * self.alphaL[T] * math.pi * distance_tmin1 * distance_tmin1 / self.area[loc]
                    P_closest_between_t_and_tmin1 = math.exp(tempTmin1 * nu_L[T][loc]) - math.exp(tempT * nu_L[T][loc])

                    #volunteer contribution between t-1 and t
                    result -= self.OHCAProbL[T] * self.lambdaL[T][loc] * P_closest_between_t_and_tmin1 * self.DeMaioSurvivalProb(t - 1)

        return result

    def gradientContDeMaio(self, nu_L) -> float:
        """
        Computes the gradient of DeMaio objective as a function of nu.

        Parameters
        ----------
        nu_L : list
            A list of volunteer **location** distribution during different
            time segments.

        Returns
        -------
        grad_L : list
            A list of gradient vector as a function of nu during different
            time segment.
        """
        grad_L = []

        for t in range(self.numTimeSeg):

            timeslot = np.arange(1, self.maxDispatchTime+1)

            # Prob (tempT and tempTmin1)
            distance_t = self.walkingSpeed / 3600 * (timeslot - self.volResDelay)
            np.clip(distance_t, 0, self.maxDispatchDistance, out=distance_t)
            tempT = np.array([-self.nVolunteers * self.alphaL[t] * math.pi * distance_t * distance_t / location_area for location_area in self.area])

            distance_tmin1 = self.walkingSpeed / 3600 * ((timeslot - 1) - self.volResDelay)
            np.clip(distance_tmin1, 0, self.maxDispatchDistance, out=distance_tmin1)
            tempTmin1 = np.array([-self.nVolunteers * self.alphaL[t] * math.pi * distance_tmin1 * distance_tmin1 / location_area for location_area in self.area])

            nu_extend = np.resize(nu_L[t], (tempT.T.shape)).T
            prob = tempTmin1 * np.exp(tempTmin1 * nu_extend) - tempT * np.exp(tempT * nu_extend)

            # DeMaioSurvivalProb
            survivalProb = self.DeMaioSurvivalProb(timeslot - 1)

            # Calculate result
            lambdaL_extend = np.resize(self.lambdaL[t], (tempT.T.shape)).T
            x = np.sum(lambdaL_extend * prob * survivalProb)

            grad = -np.sum(self.OHCAProbL[t] * lambdaL_extend * prob * survivalProb, axis=1)
            grad_L.append(grad)

        return grad_L

    #----------------------------------------------------------------
    #       Profile Waalewijn Model 1 (separable, continuous evaluation)
    #----------------------------------------------------------------
    def pWaalewijn(self, x):
        """
        Compute the negative of Waalewijn survival we want to minimize.

        Parameters
        ----------
        x : numpy.array
            An array that represents the volunteer **recruitment** distribution.

        Returns
        -------
        float
            Negative of Waalewijn survival probability with given recruitment
            distribution.
        """

        nu_L = [self.A[i] @ x for i in range(self.numTimeSeg)]
        return self.contWaalewijn(nu_L)

    def gradientPWaalewijn(self, x):
        """
        Compute the gradient of the Waalewijn objective function.

        Parameters
        ----------
        x : numpy.array
            An array that represents the volunteer **recruitment** distribution.

        Returns
        -------
        numpy.array
            Gradient of Waalewijn survival objective as a function of x.
        """

        nu_L = [self.A[i] @ x for i in range(self.numTimeSeg)]
        g = self.gradientContWaalewijn(nu_L)
        result = [self.A[i].T @ g[i] for i in range(self.numTimeSeg)]
        return sum(result)

    def evaluatePWaalewijn(self, x):
        """
        Compute the Waalewijn survival probability.

        Parameters
        ----------
        x : numpy.array
            An array that represents the volunteer **recruitment** distribution.

        Returns
        -------
        float
            Survival probability with given recruitment disbution.
        """
        return -self.pWaalewijn(x)


    def contWaalewijn(self, nu_L):
        """
        Computes the Waalewijn survival objective value we want to minimize.

        Parameters
        ----------
        nu_L : list
            A list of volunteer **location** distribution during different
            time segment.

        Returns
        -------
        float
            Negative of Waalewijn survival probability.
        """
        result = 0.0

        for T in range(self.numTimeSeg):
            for loc in range(self.nLocations):
                respTimesAmbulances = [0] * self.nAmbulances
                nAmbuFound = 0
                for i in range(0, self.nBases):
                    if (self.ambus[i] > 0):
                        count = self.ambus[i]
                        while (count > 0):
                            respTimesAmbulances[nAmbuFound] = self.travelTime_A[i][
                                loc] + self.ambuResDelay

                            count = count - 1
                            nAmbuFound = nAmbuFound + 1

                if (nAmbuFound != self.nAmbulances):
                    print('Not all ambulances found in the array ambus')

                sortedRespTime = np.array(respTimesAmbulances)
                sortedRespTime.sort()
                sortedNAmbulancesPerbase = np.zeros((self.nBases))

                count = 0
                for i in range(0, self.nAmbulances):
                    sortedNAmbulancesPerbase[count] += 1
                    if (i < self.nAmbulances - 1
                            and sortedRespTime[i + 1] != sortedRespTime[i]):
                        count += 1

                timeslot = np.arange(1, self.maxDispatchTime + 1)
                distance_ts = self.walkingSpeed / 3600 * (timeslot - self.volResDelay)
                np.clip(distance_ts, 0, self.maxDispatchDistance, out=distance_ts)

                distance_tmin1s = (self.walkingSpeed / 3600 * ((timeslot - 1) - self.volResDelay))
                np.clip(distance_tmin1s, 0, self.maxDispatchDistance, out=distance_tmin1s)

                tempTs = (-self.nVolunteers * self.alphaL[T] * math.pi * distance_ts * distance_ts / self.area[loc])
                tempTmin1s = (-self.nVolunteers * self.alphaL[T] * math.pi * distance_tmin1s * distance_tmin1s / self.area[loc])

                all_ambuCounts = np.zeros(len(np.trim_zeros(sortedNAmbulancesPerbase, 'b')) + 1)
                all_ambuCounts[1:] = np.cumsum(np.trim_zeros(sortedNAmbulancesPerbase, 'b'))
                responseTimes = np.array([sortedRespTime[int(count)] for count in all_ambuCounts[:-1]])

                probsNoAmbuBefore = self.busyProb ** all_ambuCounts
                probsAmbuAtThisBase = 1 - self.busyProb ** sortedNAmbulancesPerbase
                probsClosestAmbuAtThisBase = (probsNoAmbuBefore * probsAmbuAtThisBase[:len(all_ambuCounts)])

                probs = np.zeros_like(tempTs)
                probs = np.exp(tempTmin1s * nu_L[T][loc]) - np.exp(tempTs * nu_L[T][loc])

                #--------------------- Volunteer contribution ----------------

                # #Loop over all response times of the closest volunteer
                for t in range(1, int(self.maxDispatchTime) + 1):
                    #i closest ambulances are busy, i+1-th ambulance is first to be available
                    responseTimes_before_t = responseTimes[responseTimes < t]
                    responseTimes_on_or_after_t = responseTimes[responseTimes >= t]
                    len_before = len(responseTimes_before_t)

                    if len_before > 0:
                        result -= np.sum(self.OHCAProbL[T] * self.lambdaL[T][loc] * probs[t-1] * probsClosestAmbuAtThisBase[:len_before] * self.Waalewijn_model1(responseTimes_before_t, responseTimes_before_t))

                    result -= np.sum(self.OHCAProbL[T] * self.lambdaL[T][loc] * probs[t-1] * probsClosestAmbuAtThisBase[len_before:-1] * self.Waalewijn_model1(t-1, responseTimes_on_or_after_t))

                tempT = (- self.nVolunteers * self.alphaL[T] * math.pi * self.maxDispatchDistance * self.maxDispatchDistance / self.area[loc])
                probNoVolunteer = np.exp(tempT * nu_L[T][loc])

                result -= np.sum(self.OHCAProbL[T] * self.lambdaL[T][loc] * probNoVolunteer * probsClosestAmbuAtThisBase[:-1] * self.Waalewijn_model1(responseTimes, responseTimes))

        return result

    def gradientContWaalewijn(self, nu_L):
        """
        Computes the gradient of Waalewijn objective as a function of nu.

        Parameters
        ----------
        nu_L : list
            A list of volunteer **location** distribution during different
            time segments.

        Returns
        -------
        grad_L : list
            A list of gradient vector as a function of nu during different
            time segment.
        """
        grad_L = []

        for T in range(self.numTimeSeg):

            grad = np.zeros(self.nLocations)

            for loc in range(self.nLocations):

                respTimesAmbulances = np.zeros((self.nAmbulances))
                nAmbuFound = 0
                for i in range(0, self.nBases):
                    if (self.ambus[i] > 0):
                        count = self.ambus[i]
                        while (count > 0):
                            respTimesAmbulances[nAmbuFound] = self.travelTime_A[i][
                                loc] + self.ambuResDelay

                            count = count - 1
                            nAmbuFound = nAmbuFound + 1

                if (nAmbuFound != self.nAmbulances):
                    print('Not all ambulances found in the array ambus')

                sortedRespTime = np.array(respTimesAmbulances)
                sortedRespTime.sort()
                sortedNAmbulancesPerbase = np.zeros((self.nBases))

                count = 0
                for i in range(0, self.nAmbulances):
                    sortedNAmbulancesPerbase[count] += 1
                    if (i < self.nAmbulances - 1
                            and sortedRespTime[i + 1] != sortedRespTime[i]):
                        count += 1

                #--------------------- Volunteer contribution ----------------

                timeslot = np.arange(1, self.maxDispatchTime + 1)
                distance_ts = self.walkingSpeed / 3600 * (timeslot - self.volResDelay)
                np.clip(distance_ts, 0, self.maxDispatchDistance, out=distance_ts)
                distance_tmin1s = (self.walkingSpeed / 3600 * ((timeslot - 1) - self.volResDelay))
                np.clip(distance_tmin1s, 0, self.maxDispatchDistance, out=distance_tmin1s)

                tempTs = (-self.nVolunteers * self.alphaL[T] * math.pi * distance_ts * distance_ts / self.area[loc])
                tempTmin1s = (-self.nVolunteers * self.alphaL[T] * math.pi * distance_tmin1s * distance_tmin1s / self.area[loc])

                all_ambuCounts = np.zeros(len(np.trim_zeros(sortedNAmbulancesPerbase, 'b')) + 1)
                all_ambuCounts[1:] = np.cumsum(np.trim_zeros(sortedNAmbulancesPerbase, 'b'))
                responseTimes = np.array([sortedRespTime[int(count)] for count in all_ambuCounts[:-1]])

                probsNoAmbuBefore = self.busyProb ** all_ambuCounts
                probsAmbuAtThisBase = 1 - self.busyProb ** sortedNAmbulancesPerbase
                probsClosestAmbuAtThisBase = (probsNoAmbuBefore * probsAmbuAtThisBase[:len(all_ambuCounts)])
                probs = np.zeros_like(tempTs)

                zero_ix = distance_tmin1s == 0
                n_zero_ix = distance_tmin1s != 0

                probs[zero_ix] = (-tempTs[zero_ix] * np.exp(tempTs[zero_ix] * nu_L[T][loc]))
                probs[n_zero_ix] = (
                    tempTmin1s[n_zero_ix] *
                    np.exp(tempTmin1s[n_zero_ix] * nu_L[T][loc]) -
                    tempTs[n_zero_ix] *
                    np.exp(tempTs[n_zero_ix] * nu_L[T][loc])
                )

                #Loop over all response times of the closest volunteer
                for t in range(1, int(self.maxDispatchTime) + 1):
                    #i closest ambulances are busy, i+1-th ambulance is first to be available
                    responseTimes_before_t = responseTimes[responseTimes < t]
                    responseTimes_on_or_after_t = responseTimes[responseTimes >= t]

                    len_before = len(responseTimes_before_t)

                    if len_before:
                        grad[loc] -= np.sum(
                            self.OHCAProbL[T] *
                            self.lambdaL[T][loc] *
                            probs[t-1] *
                            probsClosestAmbuAtThisBase[:len_before] *
                            self.Waalewijn_model1(responseTimes_before_t, responseTimes_before_t)
                        )

                    grad[loc] -= np.sum(
                        self.OHCAProbL[T] *
                        self.lambdaL[T][loc] *
                        probs[t-1] *
                        probsClosestAmbuAtThisBase[len_before:-1] *
                        self.Waalewijn_model1(t - 1,responseTimes_on_or_after_t)
                    )


                #--------------------- Ambulance contribution ----------------

                #This is to account for situation without any volunteers
                tempT = (
                    - self.nVolunteers *
                    self.alphaL[T] *
                    math.pi *
                    self.maxDispatchDistance *
                    self.maxDispatchDistance /
                    self.area[loc]
                )
                probNoVolunteer = tempT * math.exp(tempT * nu_L[T][loc])

                grad[loc] -= np.sum(
                    self.OHCAProbL[T] *
                    self.lambdaL[T][loc] *
                    probNoVolunteer *
                    probsClosestAmbuAtThisBase[:-1] *
                    self.Waalewijn_model1(responseTimes, responseTimes)
                )

            grad_L.append(grad)

        return grad_L

    #----------------------------------------------------------------
    #       Profile Late Arrival
    #----------------------------------------------------------------

    def pLateArrival(self, x):
        """
        Compute the late arrival probability.

        Parameters
        ----------
        x : numpy.array
            An array that represents the volunteer **recruitment** distribution.

        Returns
        -------
        float
            Late arrival probability with given recruitment distribution.
        """
        obj = 0

        nu_L = [self.A[i] @ x for i in range(self.numTimeSeg)]
        disResTarget = (self.threshold - self.volResDelay) * self.walkingSpeed / 3600

        for t in range(self.numTimeSeg):
            for i in range(self.nLocations):
                nAmbuWithinTarget = 0

                for j in range(self.nBases):
                    if self.travelTime_A[j][i] + self.ambuResDelay <= self.threshold:
                        nAmbuWithinTarget += self.ambus[j]

                pNoAmbuAvailable = self.busyProb ** nAmbuWithinTarget

                circle = math.pi * (disResTarget ** 2)
                expo = -self.nVolunteers * self.alphaL[t] * circle * nu_L[t][i] / self.area[i]
                pNoVolunteerAvailable = np.exp(expo)

                obj += self.OHCAProbL[t] * self.lambdaL[t][i] * pNoAmbuAvailable * pNoVolunteerAvailable

        return obj

    def gradientPLate(self, x):
        """
        Compute the gradient of the late arrival objective function.

        Parameters
        ----------
        x : numpy.array
            An array that represents the volunteer **recruitment** distribution.

        Returns
        -------
        numpy.array
            Gradient of late arrival objective as a function of x.
        """

        nu_L = [self.A[i] @ x for i in range(self.numTimeSeg)]
        g = self.gradientLate(nu_L)
        result = [self.A[i].T @ g[i] for i in range(self.numTimeSeg)]
        return sum(result)

    def gradientLate(self, nu_L):
        """
        Computes the gradient of late arrival objective as a function of nu.

        Parameters
        ----------
        nu_L : list
            A list of volunteer **location** distribution during different
            time segments.

        Returns
        -------
        grad_L : list
            A list of gradient vector as a function of nu during different time segment.
        """
        grad_L = []

        for t in range(self.numTimeSeg):

            grad = np.zeros(self.nLocations)

            disResTarget = (self.threshold - self.volResDelay) * self.walkingSpeed / 3600
            for i in range(self.nLocations):

                nAmbuWithinTarget = 0

                for j in range(self.nBases):
                    if self.travelTime_A[j][i] + self.ambuResDelay <= self.threshold:
                        nAmbuWithinTarget += self.ambus[j]

                pNoAmbuAvailable = self.busyProb ** nAmbuWithinTarget

                circle = math.pi * (disResTarget ** 2)
                const = -self.nVolunteers * self.alphaL[t] * circle / self.area[i]
                grad[i] = self.OHCAProbL[t] * self.lambdaL[t][i] * pNoAmbuAvailable * const * np.exp(const * nu_L[t][i])

            grad_L.append(grad)

        return grad_L




    #----------------------------------------------------------------
    #       Maximize Waalewijn (Fix ambulance travel time)
    #----------------------------------------------------------------

    def pWaalewijn_FixEMS(self, x):
        """
        Compute the negative of Waalewijn survival with fixed EMS response time
        (13 minutes from patient collapse to ambulance arrival)  that we want
        to minimize

        Parameters
        ----------
        x : numpy.array
            An array that represents the volunteer **recruitment** distribution.

        Returns
        -------
        float
            Negative of Waalewijn survival probability with given recruitment
            distribution.
        """

        nu_L = [self.A[i] @ x for i in range(self.numTimeSeg)]
        return self.contWaalewijn_FixEMS(nu_L)

    def gradientPWaalewijn_FixEMS(self, x):
        """
        Compute the gradient of the survival with fixed EMS response time
        (13 minutes from patient collapse to ambulance arrival)

        Parameters
        ----------
        x : numpy.array
            An array that represents the volunteer **recruitment** distribution.

        Returns
        -------
        numpy.array
            Gradient of Waalewijn survival objective as a function of x.
        """

        nu_L = [self.A[i] @ x for i in range(self.numTimeSeg)]
        g = self.gradientContWaalewijn_FixEMS(nu_L)
        result = [self.A[i].T @ g[i] for i in range(self.numTimeSeg)]
        return sum(result)

    def evaluatePWaalewijn_FixEMS(self, x):
        """
        Compute the survival with fixed EMS response time
        (13 minutes from patient collapse to ambulance arrival).

        Parameters
        ----------
        x : numpy.array
            An array that represents the volunteer **recruitment** distribution.

        Returns
        -------
        float
            Survival probability with given recruitment disbution.
        """
        return -self.pWaalewijn_FixEMS(x)


    def contWaalewijn_FixEMS(self, nu_L) -> float:
        """
        Computes the survival with fixed EMS response time
        (13 minutes from patient collapse to ambulance arrival) as a function
        of nu.

        Parameters
        ----------
        nu_L : list
            A list of volunteer **location** distribution during different
            time segment.

        Returns
        -------
        float
            Negative of Waalwijn survival probability.
        """
        result = 0

        for T in range(self.numTimeSeg):
            for loc in range(self.nLocations):

                timeslot = np.arange(1, self.maxDispatchTime+1)

                distance_t = self.walkingSpeed / 3600 * (timeslot - self.volResDelay)
                np.clip(distance_t, 0, self.maxDispatchDistance, out=distance_t)
                tempT = -self.nVolunteers * self.alphaL[T] * math.pi * distance_t * distance_t / self.area[loc]

                distance_tmin1 = self.walkingSpeed / 3600 * ((timeslot - 1) - self.volResDelay)
                np.clip(distance_tmin1, 0, self.maxDispatchDistance, out=distance_tmin1)
                tempTmin1 = -self.nVolunteers * self.alphaL[T] * math.pi * distance_tmin1 * distance_tmin1 / self.area[loc]

                P_closest_between_t_and_tmin1 = np.exp(tempTmin1 * nu_L[T][loc]) - np.exp(tempT * nu_L[T][loc])

                # Assume fixed Ambulance response time
                survivalProb = self.Waalewijn_model1(timeslot - 1, 780)

                #volunteer contribution between t-1 and t
                result -= np.sum(self.OHCAProbL[T] * self.lambdaL[T][loc] * P_closest_between_t_and_tmin1 * survivalProb)

                # No volunteers
                probNoVolunteer = np.exp(-self.nVolunteers * self.alphaL[T] * math.pi * distance_t[-1] ** 2 / self.area[loc] * nu_L[T][loc])
                result -= self.OHCAProbL[T] * self.lambdaL[T][loc] * probNoVolunteer * self.Waalewijn_model1(780, 780)

        return result

    def gradientContWaalewijn_FixEMS(self, nu_L) -> float:
        """
        Computes the gradient of survival with fixed EMS response time
        (13 minutes from patient collapse to ambulance arrival) as a function of
        nu.

        Parameters
        ----------
        nu_L : list
            A list of volunteer **location** distribution during different
            time segments.

        Returns
        -------
        grad_L : list
            A list of gradient vector as a function of nu during different
            time segment.
        """
        grad_L = []

        for T in range(self.numTimeSeg):
            grad = np.zeros(self.nLocations)

            for loc in range(self.nLocations):

                timeslot = np.arange(1, self.maxDispatchTime+1)

                distance_t = self.walkingSpeed / 3600 * (timeslot - self.volResDelay)
                np.clip(distance_t, 0, self.maxDispatchDistance, out=distance_t)
                tempT = -self.nVolunteers * self.alphaL[T] * math.pi * distance_t * distance_t / self.area[loc]

                distance_tmin1 = self.walkingSpeed / 3600 * ((timeslot - 1) - self.volResDelay)
                np.clip(distance_tmin1, 0, self.maxDispatchDistance, out=distance_tmin1)
                tempTmin1 = -self.nVolunteers * self.alphaL[T] * math.pi * distance_tmin1 * distance_tmin1 / self.area[loc]

                P_closest_between_t_and_tmin1 = tempTmin1 * np.exp(tempTmin1 * nu_L[T][loc]) - tempT * np.exp(tempT * nu_L[T][loc])

                # Assume fixed Ambulance response time
                survivalProb = self.Waalewijn_model1(timeslot - 1, 780)

                #volunteer contribution between t-1 and t
                grad[loc] -= np.sum(self.OHCAProbL[T] * self.lambdaL[T][loc] * P_closest_between_t_and_tmin1 * survivalProb)

                # No volunteers
                tempNoVol = -self.nVolunteers * self.alphaL[T] * math.pi * distance_t[-1] ** 2 / self.area[loc]
                probNoVolunteer = tempNoVol * np.exp(tempNoVol * nu_L[T][loc])
                grad[loc] -= self.OHCAProbL[T] * self.lambdaL[T][loc] * probNoVolunteer * self.Waalewijn_model1(780, 780)

            grad_L.append(grad)


        return grad_L


    #----------------------------------------------------------------
    #       Algorithm
    #----------------------------------------------------------------

    def AdaptiveLineSearch(self, alpha, val, x, g, d, theta=0.2, gamma=0.7, verbose=False, method='d'):
        """
        Adaptive line search algorithm that determines the step size for
        Frank-Wolfe algorithm.

        Parameters
        ----------
        alpha : float
            Stepsize obtained from previous iteration.
        val : float
            Current objective value.
        x : numpy.array
            Current iterate.
        g : numpy.array
            Gradient.
        d : numpy.array
            Search direction.
        theta : float, default=0.2
            Parameter for adaptive line search algorithm.
        gamma : float, default=0.7
            Parameter for adaptive line search algorithm.
        verbose : boolean, default=False
            A flag indicates whether we want to print information mainly for the
            purpose of debugging.
        method : {'l', 'd', 'w', 'w2'}, default='d'
            Indicate which objective function we are using. 'l' - late arrival
            probability, 'd' - DeMaio, 'w' - Waalewijn, 'w2' - Waalewijn with
            fixed EMS response.

        Returns
        -------
        alpha_k : float
            An appropriate stepsize that satisfies the sufficient decrease
            condition.
        """
        const = np.dot(g, d)

        while True:
            x_new = x + alpha * d

            if method == 'd':
                new_val = self.pDeMaio(x_new)
            elif method == 'w':
                new_val = self.pWaalewijn(x_new)
            elif method == 'w2':
                new_val = self.pWaalewijn_FixEMS(x_new)
            else:
                new_val = self.pLateArrival(x_new)

            # Check for sufficient decrease:
            RHS = val + alpha * theta * const

            if verbose:
                print(val, new_val, RHS, alpha)

            if new_val <= RHS:
                alpha_k = alpha
                return alpha_k

            alpha = alpha * gamma



    def Frankwolfe_LB(self, init_x, eps=1e-3, tol=5e-3, theta=0.2, gamma=0.7, verbose=False, method='d'):
        """
        A modified version of Away-step Frank-Wolfe algorithm.

        Parameters
        ----------
        init_x : numpy.array
            An initial value $x_0$.
        eps : float, default=1e-3
            Stopping criteria for gradient.
        tol : float, default=5e-3
            Stopping criteria when objective function gets close enought to
            the lower bound.
        theta : float, default=0.2
            Parameter for adaptive line search algorithm.
        gamma : float, default = 0.7
            Parameters for adaptive line search algorithm.
        verbose : boolean, default=False
            A flag indicates whether we want to print information mainly for the
            purpose of debugging.
        method : {'l', 'd', 'w', 'w2'}, default='d'
            Indicate which objective function we are using. 'l' - late arrival
            probability, 'd' - DeMaio, 'w' - Waalewijn, 'w2' - Waalewijn with
            fixed EMS response.

        Returns
        -------
        x : numpy.array
            Optimal volunteer recruitment distribution.
        current_iter : list
            A list of objective values obtained during each iteration of the
            algorithm.
        lower_bound : list
            A list of lower bound on objective function that computed during
            each iteration of the algorithm.
        """
        x = np.array(init_x)
        Dict = []

        current_iter = []
        lower_bound = []
        alpha_L = [1]

        k = 1
        while True:

            if method == 'd':
                grad_k = self.gradientPDeMaio(x)
            elif method == 'w':
                grad_k = self.gradientPWaalewijn(x)
            elif method == 'w2':
                grad_k = self.gradientPWaalewijn_FixEMS(x)
            else:
                grad_k = self.gradientPLate(x)

            i = np.argmin(grad_k)
            s_k = self.standard(self.nLocations, i)
            d_FW = s_k - x
            g_FW = np.dot(-grad_k, d_FW)

            # Added lower_bound
            if method == 'd':
                val = self.pDeMaio(x)
            elif method == 'w':
                val = self.pWaalewijn(x)
            elif method == 'w2':
                val = self.pWaalewijn_FixEMS(x)
            else:
                val = self.pLateArrival(x)

            Dict.append((grad_k.copy(), x.copy(), val))
            lower = self.getLowerbound(Dict)
            current_iter.append(val)
            lower_bound.append(lower)

            if g_FW <= eps:
                break

            if abs(val-lower) <= tol:
                print("close enough to lower bound")
                break

            S_k = [i for i, e in enumerate(x) if e > 1e-12]
            sub_grad = [grad_k[i] for i in S_k]
            j = S_k[np.argmax(sub_grad)]
            v_k = self.standard(self.nLocations, j)
            d_A = x - v_k
            g_A = np.dot(-grad_k, d_A)

            if g_FW >= g_A:
                d_k = d_FW
                alpha_max = 1
            else:
                if verbose:
                    print("Away step!")
                d_k = d_A
                alpha_max = x[j] / (1 - x[j])


            D = {}
            alpha = min(alpha_max, alpha_L[-1])
            alpha_k = self.AdaptiveLineSearch(alpha, val, x, grad_k, d_k, theta=theta, gamma=gamma, method=method)
            alpha_L.append(alpha_k / gamma)

            x += alpha_k * d_k

            if verbose:
                print("Iteration", k, val, lower, sum(x), g_FW,
                      alpha_k)
                k += 1

        return x, current_iter, lower_bound
