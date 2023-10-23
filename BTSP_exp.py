# Author @Etienne Barou

# This code is to experiment with the ratinabox library and to simulate the environment of the BTSP papers 

# All plotting functions return a tuple (fig, ax) of matplotlib figure objects.

import numpy as np
import matplotlib.pyplot as plt 
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, GridCells, BoundaryVectorCells
from tqdm.notebook import tqdm  # gives time bar

import Pyramidal_neurons
from Pyramidal_neurons import theta_gating

# Define the environment which is a linear treadmill with boundaries on the side but not at the top or the bottom
# The animal executes a lap when it comes to the end of the track and restarts at the bottom

Env = Environment(params={
    'boundary_conditions' : 'periodic', # No boundary conditions on the sides 
    'scale' : 2, # scale of environment (in metres)
    'boundary' : [[1.2,2], [1.2,0], [0.8,0],[0.8,2]],
    'walls' : [[[0.8, 0], [0.8,2]], [[1.2,0],[1.2,2]]]
})


# Define the agent
Ag = Agent(Env, params={
    #"thigmotaxis": 0,
    #"rotational_velocity_std": 0
})

# Making our own trajectory to make sure the rat goes in a straight line on the treadmill

# Initiate starting position to always be the same 
Ag.pos = [float(1),float(0)]

# Define the x and y coordinates of the position of the rat 
times = list(range(1, 101, 1))
positions = np.linspace([1,0],[1,2], 100).tolist()

# Import the trajectory
Ag.import_trajectory(times=times, positions=positions, interpolate=False)

    
# Define mice speed

# Slow lap (as per research paper) 
Ag.speed_mean = 0.12 # cm/s

# Fast lap (as per research paper)
# Ag.speed_mean = 0.3 # cm/s


# Initialize number of cells to 26, as per research papers 
n_cells = 26 

place_cells_pos = np.linspace([1,0],[1,2], 26) # Defining positions for place cells to be uniform

# Initializing place cells 
PCs = PlaceCells(Ag, params={
'name' : 'PlaceCells',
'n': n_cells, # Controlling the number of place cells 
'widths': 0.09, # controlling the width of the place cell in meters (m) / width of rate time activity 
"place_cell_centres": place_cells_pos,
"wall_geometry": "line_of_sight",
})

print(PCs.get_state())
PCs_firing_rate = PCs.get_state()

# Define Pyramidal Neurons 
PyramidalNeurons = Pyramidal_neurons.PyramidalNeurons(Ag, params={'n': n_cells, 'name':"Pyramidal Neurons"})

# Adding basal input (place cells) to pyramidal cells 
#PyramidalNeurons.basal_compartment.add_input(PCs) # eta = learning rate of the weights 

# Adding the place cell locations as input for the basal compartment of the pyramidal neurons
# PyramidalNeurons.basal_compartment.inputs['PlaceCells']['I'] = PCs_firing_rate


# Train the model
# 6 Simulate.
dt = 50e-3 
T = 3 * 60 #3 mins / 36 laps  

lap_counter = 0

for i in tqdm(range(int(T / dt))):
    # Update Agent 
    Ag.update(dt=dt)

    # Update Place cells 
    PCs.update()

    # Update Pyramidal Neurons 
    PyramidalNeurons.update_dendritic_compartments()
    PyramidalNeurons.update()
    PyramidalNeurons.update_weights()
    # Count number of laps 
    if Ag.pos[0] == 1 and Ag.pos[1] == 2: 
        lap_counter += 1

print("Number of laps : ", lap_counter)
 
# plot trajectory.
Ag.plot_trajectory() # Plot the trajectory

# Plot the timeseries
PCs.plot_rate_timeseries()

# PyramidalNeurons.plot_rate_map()
# PyramidalNeurons.plot_rate_timeseries()

# First we plot the loss. This is a (smoothed) trace of the mean absolute difference between 
# the voltage in the basal compartment (the ground truth) and the voltage in the apical compartment 
# (the compartment learning to replicate the ground truth using recurrent and conjunctive velocity inputs).
# PyramidalNeurons.plot_loss()

# Plot place cell locations.
PCs.plot_place_cell_locations()


plt.show()

