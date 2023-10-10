# Author @Etienne Barou

# This code is to experiment with the ratinabox library and to simulate the environment of the BTSP papers 

# All plotting functions return a tuple (fig, ax) of matplotlib figure objects.

import numpy as np
import matplotlib.pyplot as plt 
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, GridCells, BoundaryVectorCells
from tqdm.notebook import tqdm  # gives time bar



def simple_test():

    # Define the environment, agent and cells
    
    """
    # Classic Circular environment 
    # Need to figure out how to make the rat run on a single line in a circular motion 
    Env = Environment(params = {
    'boundary':[[0.5*np.cos(t),0.5*np.sin(t)] for t in np.linspace(0,2*np.pi,100)],
    })
    """

    # Circular treadmill
    Env = Environment(params={
        'boundary':[[0.5*np.cos(t),0.5*np.sin(t)] for t in np.linspace(0,2*np.pi,100)],
        
        # By changing the constants in front of the two coordinates you can change the width of the treadmill
        'holes' : [[[0.43*np.cos(t),0.43*np.sin(t)] for t in np.linspace(0,2*np.pi,100)]]
    })

    Ag = Agent(Env)
    
    # Slow lap (as per research paper) 
    Ag.speed_mean = 0.12 # cm/s

    # Fast lap (as per research paper)
    Ag.speed_mean = 0.3 # cm/s

    # Adding circular motion
    Ag.rotation_velocity_std = 120 * np.pi/180 #radians 
    Ag.rotational_velocity_coherence_time = 0.08
   
    # Initializing place cells 
    PCs = PlaceCells(Ag,params={
    'noise_std':0.2, #defaults to 0 i.e. no noise
    'noise_coherence_time':0.5, #autocorrelation timescale of additive noise vector 
    "wall_geometry": "line_of_sight",
})

   # 6 Simulate.
    dt = 50e-3
    T = 5 * 60 #5 mins  

    for i in tqdm(range(int(T / dt))):
        Ag.update(dt=dt)
        PCs.update()

    # 7 Plot trajectory.
    fig, ax = Ag.plot_position_heatmap()
    fig, ax = Ag.plot_trajectory(t_start=Ag.t-30, fig=fig, ax=ax,color="changing")
    
    # Plot the timeseries
    fig, ax = PCs.plot_rate_timeseries()
    
    # 9 Plot place cell locations.
    fig, ax = PCs.plot_place_cell_locations()

    # 10 Plot rate maps (first analytically, second using bin exploration data, third using observed spikes ) .
    #fig, ax = PCs.plot_rate_map(chosen_neurons="3", method="groundtruth")
    #fig, ax = PCs.plot_rate_map(chosen_neurons="3", method="history")
    #fig, ax = PCs.plot_rate_map(chosen_neurons="3", method="neither", spikes=True)
    
    plt.show()
    
    return

simple_test()
