# Author @Etienne Barou

# This code is to experiment with the ratinabox library and to simulate the environment of the BTSP papers 

import numpy as np
import matplotlib.pyplot as plt 
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, GridCells, BoundaryVectorCells


def simple_test():

    # Define the environment, agent and cells
    Env = Environment()
    Ag = Agent(Env)
    PCs = PlaceCells(Ag)

    # Loop to update the agent position and the Place cells firing 
    for i in range(int(60/Ag.dt)):
        Ag.update() # updates velocity based on movement policy
        PCs.update()

    print("Timestamps:", Ag.history['t'][:10], "\n")
    print("Positions:",Ag.history['pos'][:10],"\n")
    print("Firing rate timeseries:",PCs.history['firingrate'][:10],"\n")
    print("Spikes:",PCs.history['spikes'][:10], "\n")
    print("I changed this")

    # Plot the timeseries 
    Ag.plot_trajectory()
    PCs.plot_rate_timeseries()

    plt.show()
    
    return

simple_test()
