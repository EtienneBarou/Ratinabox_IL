# Author @Etienne Barou

# This code is to experiment with the ratinabox library and to simulate the environment of the BTSP papers 

# All plotting functions return a tuple (fig, ax) of matplotlib figure objects.

import numpy as np
import ratinabox
import matplotlib.pyplot as plt 
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, GridCells, BoundaryVectorCells
from tqdm.notebook import tqdm  # gives time bar
ratinabox.autosave_plots=True; ratinabox.figure_directory = "../Ratinabox_IL/figures/"


import Pyramidal_neurons

    # Define the environment which is a linear treadmill with boundaries on the side but not at the top or the bottom
    # The animal executes a lap when it comes to the end of the track and restarts at the bottom

def BTSP_data():
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

    # Start like this so the place cells don't overflow in a direction or the other
    place_cells_pos = np.linspace([1,0.16],[1,1.84], 26) # Defining positions for place cells to be uniform

    # Initializing place cells 
    PCs = PlaceCells(Ag, params={
    'name' : 'PlaceCells',
    'n': n_cells, # Controlling the number of place cells 
    'widths': 0.09, # controlling the width of the place cell in meters (m) / width of rate time activity 
    "place_cell_centres": place_cells_pos,
    "wall_geometry": "line_of_sight",
    })

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
    Ag.plot_trajectory() 

    # Plot the timeseries
    PCs.plot_rate_timeseries()

    # Plot the rate map
    PCs.plot_rate_map()

    # Agent x positions 
    x_position = []
    for pos in Ag.positions:
        x_position.append(pos[1])

    # Tuning centers for place cells 
    PCs_centers = [] 
    for center in PCs.place_cell_centres:
        PCs_centers.append(center[1])
    
    """
    # Plot the trajectory.
    Ag.plot_trajectory() 
    # Plot the timeseries
    PCs.plot_rate_timeseries()
    # Plot place cell locations.
    PCs.plot_place_cell_locations()
    """
    
    return x_position, PCs_centers

BTSP_data()