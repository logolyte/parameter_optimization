#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parameter_optimization.py

Functions for parameter optimization. Has a function for calculating mean squared error and functions for performing
particle swarm optimization and simulated annealing.

Author: Love Sundin
Date: 2025-06-11
"""

import numpy

def mean_squared_error(true_values, predicted_values, normalize=False, lower_bound=5e-2):
    """
    Calculates the mean squared error of predictions.

    Parameters
    ----------
    true_values : array_like
        The true values to compare against.
    predicted_values : array_like
        The predicted values.
    normalize : bool, optional
        Whether or not to apply normalization by calculating the error relative to the average along
        the last axis. The default is False.
    lower_bound : float, optional
        Lower bound of concentrations used during normalization. Values below the lower bound are
        set to the lower bound. The default is 5e-2.

    Returns
    -------
    float
        Mean squared error.

    """
    if normalize:
        average_values = numpy.clip(numpy.mean(true_values, axis=-1), a_min=lower_bound, a_max=None)
        estimated_norm = (predicted_values.T / average_values).T
        true_norm = (true_values.T / average_values).T
        return numpy.square(numpy.subtract(true_norm, estimated_norm)).mean()
    else:
        return numpy.square(numpy.subtract(true_values, predicted_values)).mean()

def particle_swarm(cost_function, bounds, particles=20, c1 = 0.8, c2 = 0.3, w = 1, iterations = 150):
    """
    Performs particle swarm optimization.

    Parameters
    ----------
    cost_function : function
        Cost function applied to the parameters. Takes all parameters as the argument and returns the
        cost as a single value.
    bounds : array_like
        Matrix with dimensions (2, dimensions) with lower and upper bounds for all parameters.
    particles : int, optional
        Number of particles used. The default is 20.
    c1 : float, optional
        Cognitive coefficient. The default is 0.8.
    c2 : float, optional
        Social coefficient. The default is 0.3.
    w : float, optional
        Inertia weight constant. The default is 1.
    iterations : int, optional
        Number of iterations to run for. The default is 150.

    Returns
    -------
    global_best_cost : float
        The lowest cost found.
    global_best_position : numpy.ndarray
        The parameter values with the lowest cost.

    """
    
    dimensions = len(bounds[0])
    lower_bounds = bounds[0]
    upper_bounds = bounds[1]
    
    # Initialize particles
    position_array = numpy.random.uniform(lower_bounds, upper_bounds, (particles, dimensions))
    velocity_array = numpy.zeros((particles, dimensions))
    
    best_positions = numpy.copy(position_array)
    best_costs = numpy.array([cost_function(position) for position in position_array])
    global_best_position = best_positions[numpy.argmin(best_costs)]
    global_best_cost = numpy.min(best_costs)
    
    # Iterate
    for iteration in range(iterations):
        print(f"\rIteration {iteration+1}/{iterations}\tLowest cost: {global_best_cost:.4e}", end="")
        
        r1 = numpy.random.uniform(0, 1, (particles, dimensions))
        r2 = numpy.random.uniform(0, 1, (particles, dimensions))
        
        # Update positions and velocities
        position_array = numpy.clip(position_array+velocity_array, a_min=lower_bounds, a_max=upper_bounds)
        velocity_array = w*velocity_array + c1*r1*(best_positions-position_array) + c2*r2*(global_best_position-position_array)
        
        # Update particle bests
        cost_array = numpy.array([cost_function(position) for position in position_array])
        improved_indices = numpy.where(cost_array < best_costs)
        best_positions[improved_indices] = position_array[improved_indices]
        best_costs[improved_indices] = cost_array[improved_indices]
        
        # Update global best
        iteration_best_cost = numpy.min(cost_array)
        if iteration_best_cost < global_best_cost:
            global_best_position = position_array[numpy.argmin(cost_array)]
            global_best_cost = iteration_best_cost
    
    print()
    
    return global_best_cost, global_best_position

def simulated_annealing(cost_function, bounds, starting_temperature = 1, step_size = 0.1, iterations = 2000, print_interval=10):
    """
    Performs simulated annealing to find optimal parameters. The cooling schedule used is
    temperature = starting_temperature / (1 + iteration/10).

    Parameters
    ----------
    cost_function : function
        The cost function to minimize.
    bounds : array_like
        Matrix with dimensions (2, dimensions) with lower and upper bounds for all parameters.
    starting_temperature : float, optional
        Starting temperature. The default is 1.
    step_size : float, optional
        How much parameters are changed at most when a step is taken. The default is 0.1.
    iterations : int, optional
        Number of iterations to run for. The default is 2000.
    print_interval : int, optional
        How many steps to take before printing the current progress. The default is 10.

    Returns
    -------
    best_score : float
        The lowest score found.
    best_position : numpy.ndarray
        Parameter values giving the lowest cost.

    """
    dimensions = len(bounds[0])
    lower_bounds = bounds[0]
    upper_bounds = bounds[1]
    interval_sizes = upper_bounds-lower_bounds
    print_indices = numpy.zeros(iterations, dtype=bool)
    print_indices[::print_interval]=True
    
    # Generate starting position
    current_position = numpy.random.uniform(lower_bounds, upper_bounds, (dimensions))
    current_score = cost_function(current_position)
    
    best_position, best_score = current_position, current_score
    
    # Iterate
    for iteration, print_index in zip(range(iterations), print_indices):
        if print_index:
            print(f"\rIteration {iteration+1}/{iterations}\tCurrent cost:{current_score:.4e}\tLowest cost: {best_score:.4e}", end="")
    
        # Update temperature
        temperature = starting_temperature / (1 + iteration/10)
        
        # Get neighbor
        neighbor = current_position.copy()
        changed_index = numpy.random.randint(dimensions-1)
        neighbor[changed_index] += numpy.random.uniform(-step_size, step_size)*interval_sizes[changed_index]
        neighbor = numpy.clip(neighbor, a_min = lower_bounds, a_max = upper_bounds)
        neighbor_score = cost_function(neighbor)
        
        # Check if change should be accepted
        if neighbor_score < current_score or numpy.random.random() < numpy.exp((current_score-neighbor_score) / temperature):
            current_position, current_score = neighbor, neighbor_score
            
            # Update the best score
            if current_score < best_score:
                best_position, best_score = current_position, current_score
    
    print()
    
    return best_score, best_position