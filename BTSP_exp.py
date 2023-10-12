# Author @Etienne Barou

# This code is to experiment with the ratinabox library and to simulate the environment of the BTSP papers 

# All plotting functions return a tuple (fig, ax) of matplotlib figure objects.

import numpy as np
import matplotlib.pyplot as plt 
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, GridCells, BoundaryVectorCells
from tqdm.notebook import tqdm  # gives time bar

"""

# Classic Circular environment 
# Need to figure out how to make the rat run on a single line in a circular motion 
Env = Environment(params = {
'boundary':[[0.5*np.cos(t),0.5*np.sin(t)] for t in np.linspace(0,2*np.pi,100)],
})

"""

# Circular treadmill Environment
Env = Environment(params={
    'boundary':[[0.5*np.cos(t),0.5*np.sin(t)] for t in np.linspace(0,2*np.pi,100)],
        
    # By changing the constants in front of the two coordinates you can change the width of the treadmill
    'holes' : [[[0.43*np.cos(t),0.43*np.sin(t)] for t in np.linspace(0,2*np.pi,100)]]
})

# Define the agent
Ag = Agent(Env)
    
# Define mice speed 
# Slow lap (as per research paper) 
Ag.speed_mean = 0.12 # cm/s

# Fast lap (as per research paper)
Ag.speed_mean = 0.3 # cm/s

# Adding circular motion to make sure the rat runs around the maze 
Ag.rotation_velocity_std = 120 * np.pi/180 #radians 
#Ag.rotational_velocity_coherence_time = 0.08
   
# Initializing place cells 
PCs = PlaceCells(Ag,params={
'noise_std':0, #defaults to 0 i.e. no noise
'noise_coherence_time':0.5, #autocorrelation timescale of additive noise vector 
"wall_geometry": "line_of_sight",
})

# 6 Simulate.
dt = 50e-3 
T = 5 * 60 #5 mins  

for i in tqdm(range(int(T / dt))):
    Ag.update(dt=dt)
    PCs.update()

# lot trajectory.
Ag.plot_trajectory(t_start=Ag.t-30) # Plot the last 30 seconds of the trajectory

# Plot the timeseries
PCs.plot_rate_timeseries()

# Plot place cell locations.
PCs.plot_place_cell_locations()

# Plot rate maps (first analytically, second using bin exploration data, third using observed spikes ) .
#PCs.plot_rate_map(chosen_neurons="3", method="groundtruth")
#PCs.plot_rate_map(chosen_neurons="3", method="history")
#PCs.plot_rate_map(chosen_neurons="3", method="neither", spikes=True)

plt.show()