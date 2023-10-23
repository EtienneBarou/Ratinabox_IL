# Author @Etienne Barou

# This code is to experiment with the ratinabox library and to simulate the environment of the BTSP papers 

# All plotting functions return a tuple (fig, ax) of matplotlib figure objects.

import numpy as np
import matplotlib.pyplot as plt 
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, GridCells, BoundaryVectorCells
from tqdm.notebook import tqdm  # gives time bar
import Theta_Seq,  Pyramidal_neurons
from tempfile import TemporaryFile
"""

# Classic Circular environment 
# Need to figure out how to make the rat run on a single line in a circular motion 
Env = Environment(params = {
'boundary':[[0.5*np.cos(t),0.5*np.sin(t)] for t in np.linspace(0,2*np.pi,100)],
})

"""

"""
# Circular treadmill Environment
Env = Environment(params={
    'boundary':[[0.5*np.cos(t),0.5*np.sin(t)] for t in np.linspace(0,2*np.pi,100)],
        
    # By changing the constants in front of the two coordinates you can change the width of the treadmill
    'holes' : [[[0.45*np.cos(t),0.45*np.sin(t)] for t in np.linspace(0,2*np.pi,100)]]
})
"""

# Linear environment with no boundary conditions in a straight line 
Env = Environment(params={
    'boundary_conditions' : 'periodic',
    'scale' : 2, # scale of environment (in metres)
    'boundary' : [[1.2,2], [1.2,0], [0.8,0],[0.8,2]],
    'walls' : [[[0.8, 0], [0.8,2]], [[1.2,0],[1.2,2]]]
})


# Define the agent
Ag = Agent(Env, params={
    "thigmotaxis": 0,
    "rotational_velocity_std": 0
})

# Change starting position to always be the same 
Ag.pos = [float(1),float(0)]

outfile = TemporaryFile()

times = list(range(1, 101, 1))
print(times)
positions = np.linspace([1,0],[1,2], 100).tolist()
print(positions)
saved_file = np.savez(outfile, times, positions)


Ag.import_trajectory(times=times, positions=positions, interpolate=False)

    
# Define mice speed 
# Slow lap (as per research paper) 
Ag.speed_mean = 0.12 # cm/s

# Fast lap (as per research paper)
# Ag.speed_mean = 0.3 # cm/s

# Adding circular motion to make sure the rat runs around the maze 
# Ag.rotation_velocity_std = 0 #radians 
# Ag.rotational_velocity_coherence_time = 0.08


# Initialize number of cells 
n_cells = 26

# Initializing place cells 
PCs = PlaceCells(Ag, params={
'n': n_cells, # Controlling the number of place cells 
'widths': 0.09, # controlling the width of the place cell in meters (m) / width of rate time activity 
'noise_std': 0, # defaults to 0 i.e. no noise
'noise_coherence_time': 0.5, # autocorrelation timescale of additive noise vector 
"wall_geometry": "line_of_sight",
})

# Define Pyramidal Neurons 
PyramidalNeurons = Pyramidal_neurons.PyramidalNeurons(Ag, params={'n': n_cells, 'name':"Pyramidal Neurons"})

# 6 Simulate.
dt = 50e-3 
T = 5 * 60 #5 mins  

for i in tqdm(range(int(T / dt))):
    Ag.update(dt=dt)
    print(i, Ag.pos)
    PCs.update()

# TODO 
# Figure out the width of the place cells 
# Making sure the mice runs in a circular motion always
# Uniform place cell repartition along the track
# Input specific length and width of track
# theta oscillation
# look at how we can decode neuronal data from generated data to reuse in the model 

# plot trajectory.
Ag.plot_trajectory() # Plot the first 30 seconds of the trajectory

# Use the theta oscillation 
#Theta_Seq.ThetaSequenceAgent(Ag)

# Plot the timeseries
PCs.plot_rate_timeseries()

# Plot place cell locations.
PCs.plot_place_cell_locations()

# Plot rate maps (first analytically, second using bin exploration data, third using observed spikes ) .
#PCs.plot_rate_map(chosen_neurons="3", method="groundtruth")
#PCs.plot_rate_map(chosen_neurons="3", method="history")
#PCs.plot_rate_map(chosen_neurons="3", method="neither", spikes=True)

plt.show()

