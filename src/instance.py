"""
instance.py
=========================================

Module that used to read inputFiles and store information of a problen instance.
"""

import sys


class Instance(object):
    """
    Read input files and store information of a problen instance.

    Args
    ----
    pop_dir : str
        Path of input population file for the region of interest.
    travelA_dir : str
        Path of the file that stores ambulance travel time information.
    nVolunteers : int
        Number of volunteers.
    volResDelay : float
        Volunteer response time delay in seconds from patient collapse.
    volWalkingSpeed : float
        Volunteer constance walking speed.
    nBases : int
        Number of ambulance bases.
    ambuResDelay : float
        Ambulance response time delay in seconds from patient collapse.
    ambuBusyProb : float
        Probability that an ambulance will be busy at time of the call.
    ambuDist : list
        Ambulance distribution over the set of bases.
    threshold : float
        Response time target.
    maxDispatchDistance : float
        Dispatch radius in kilometer within which volunteers will be alerted.
    nSteps : int
        Number of steps in greedy algorithm.

    Attributes
    ----------
    nVolunteers : int
        Number of volunteers.
    volResDelay : float
        Volunteer response time delay in seconds from patient collapse.
    volWalkingSpeed : float
        Volunteer constance walking speed.
    nBases : int
        Number of ambulance bases.
    ambuResDelay : float
        Ambulance response time delay in seconds from patient collapse.
    ambuBusyProb : float
        Probability that an ambulance will be busy at time of the call.
    ambuDist : list
        Ambulance distribution over the set of bases.
    threshold : float
        Response time target.
    maxDispatchDistance : float
        Dispatch radius in kilometer within which volunteers will be alerted.
    nSteps : int
        Number of steps in greedy algorithm.
    nLocations : int
        Number of locations.
    postalCode : list
        List of area unit code for each region.
    population : list
        List of population for each region.
    area : list
        List of area for each region.
    travelTime_A : list
        Travel time for ambulance from bases to each location.
    """
    def __init__(self, pop_dir: str, travelA_dir: str,
                 nVolunteers: int, volResDelay: float, volWalkingSpeed: float,
                 nBases: int, ambuResDelay: float, ambuBusyProb: float, ambuDist: list,
                 threshold: float, maxDispatchDistance: float, nSteps: int) -> None:

        # Read from input files
        self.readPop(pop_dir)
        self.readTravel(travelA_dir)

        # Set other parameters
        self.nVolunteers = nVolunteers
        self.volResDelay = volResDelay
        self.volWalkingSpeed = volWalkingSpeed
        self.nBases = nBases
        self.ambuResDelay = ambuResDelay
        self.ambuBusyProb = ambuBusyProb
        self.threshold = threshold
        self.maxDispatchDistance = maxDispatchDistance
        self.nSteps = nSteps

        # Set ambulance distribution
        if len(ambuDist) != self.nBases:
            print('No correct ambulance distribution specified in instance. I will assume no ambulances')
            self.ambuDist = [0] * self.nBases
        else:
            self.ambuDist = ambuDist

    def readPop(self, pop: str) -> None:
        """
        Read information from population file.

        Parameters
        ----------
        pop : str
            Path of input population file for the region of interest.
        """
        try:
            f = open(pop, 'r')
        except:
            message = "This input file '{} does not exist.".format(pop)
            sys.exit(message)

        L = [line.split() for line in f][1:]
        self.nLocations = len(L)
        self.postalCode = [int(node[0]) for node in L]
        self.population = [int(node[1]) for node in L]
        self.area = [float(node[2]) for node in L]
        f.close()

    def readTravel(self, travel_A: str) -> None:
        """
        Read travel time information.

        Parameters
        ----------
        travel_A : str
            Path of the file that stores ambulance travel time information.
        """
        try:
            f1 = open(travel_A, 'r')
        except:
            message = "This input file '{} does not exist.".format(travel_A)
            sys.exit(message)

        self.travelTime_A = [list(map(int, line.split())) for line in f1]
        f1.close()
