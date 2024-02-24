'''
CSCI 4302/5302 Advanced Robotics
University of Colorado Boulder
(C) 2023 Prof. Bradley Hayes <bradley.hayes@colorado.edu>
v1.0
'''

from __future__ import annotations
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import math
import random

###############################################################################
## Base Code - Nothing particularly interesting in here
###############################################################################

class Node:
    """
    Node for RRT/RRT* Algorithm. This is what you'll make your graph with!
    """
    def __init__(self, point: np.ndarray, parent: Node=None):
        self.point = point # n-Dimensional point
        self.parent = parent # Parent node
        self.path_from_parent = [] # List of points along the way from the parent node (for visualization)
        self.path_length = 0 # length of path from root to this node
        self.children = [] # List of child nodes

def get_nd_obstacle(state_bounds: np.ndarray) -> list:
    """
    Helper function to create an n-dimensional obstacle given the allowable bounds of each dimension
    """
    center_vector = []
    for d in range(state_bounds.shape[0]):
        center_vector.append(state_bounds[d][0] + random.random()*(state_bounds[d][1]-state_bounds[d][0]))
    radius = random.random() * 0.6
    return [np.array(center_vector), radius]

def setup_random_2d_world():
    state_bounds = np.array([[0,10],[0,10]])

    obstacles = [] # [pt: np.ndarray, radius: float] circular obstacles
    for n in range(30):
        obstacles.append(get_nd_obstacle(state_bounds))

    def state_is_valid(state: np.ndarray) -> bool:
        for dim in range(min(state.shape[0], state_bounds.shape[0])):
            # Make sure values for each dimension of the given state is within allowable bounds
            if state[dim] < state_bounds[dim][0]: return False
            if state[dim] >= state_bounds[dim][1]: return False
        for obs in obstacles:
            # Only check first n-dimensions of state for n-dimensional obstacles
            # Discard if in collision with another obstacle
            if np.linalg.norm(state[:len(obs[0])] - obs[0]) <= obs[1]: return False
        return True

    return state_bounds, obstacles, state_is_valid

def setup_fixed_test_2d_world():
    '''
    Use a deterministic 2D world for initial result consistency
    '''
    state_bounds = np.array([[0,1],[0,1]])
    obstacles = [] # [pt: np.ndarray, radius: float] circular obstacles
    obstacles.append([[0.1,0.5],0.2])
    obstacles.append([[0.2,0.05],0.2])
    obstacles.append([[0.1,0.7],0.1])
    obstacles.append([[0.7,0.2],0.1])
    obstacles.append([[0.7,0.7],0.2])

    def state_is_valid(state):
        for dim in range(min(state.shape[0], state_bounds.shape[0])):
            if state[dim] < state_bounds[dim][0]: return False
            if state[dim] >= state_bounds[dim][1]: return False
        for obs in obstacles:
            # Only check first n-dimensions of state for n-dimensional obstacles
            if np.linalg.norm(state[:len(obs[0])] - obs[0]) <= obs[1]: return False
        return True

    return state_bounds, obstacles, state_is_valid


def plot_circle(x: float, y: float, radius: float, color: str="-k"):
    deg = np.linspace(0,360,50)

    xl = [x + radius * math.cos(np.deg2rad(d)) for d in deg]
    yl = [y + radius * math.sin(np.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color)

def visualize_2D_graph(state_bounds: np.ndarray, obstacles: list, nodes: list, goal_point: np.ndarray=None, filename: str=None):
    '''
    @param state_bounds Array of min/max for each dimension
    @param obstacles Locations and radii of spheroid obstacles
    @param nodes List of vertex locations
    @param edges List of vertex connections
    '''

    fig = plt.figure()
    plt.xlim(state_bounds[0,0], state_bounds[0,1])
    plt.ylim(state_bounds[1,0], state_bounds[1,1])

    for obs in obstacles:
        plot_circle(obs[0][0], obs[0][1], obs[1])

    goal_node = None
    for node in nodes:
        if node.parent is not None:
            node_path = np.array(node.path_from_parent)
            plt.plot(node_path[:,0], node_path[:,1], '-g')
        if goal_point is not None and np.linalg.norm(node.point - goal_point) <= 1e-2:
            goal_node = node
            plt.plot(node.point[0], node.point[1], 'k^')
        else:
            plt.plot(node.point[0], node.point[1], 'ro')

    if len(nodes) > 0:
        plt.plot(nodes[0].point[0], nodes[0].point[1], 'ko')

        if goal_node is not None:
            cur_node = goal_node
            visited_nodes = []
            while cur_node is not None: 
                if cur_node in visited_nodes:
                    print("ERROR: Cycle found in path_from_parent trace.")
                    break
                if cur_node.parent is not None:
                    node_path = np.array(cur_node.path_from_parent)
                    plt.plot(node_path[:,0], node_path[:,1], '--y')
                    visited_nodes.append(cur_node)
                    cur_node = cur_node.parent
                else:
                    break

    if goal_point is not None:
        plt.plot(goal_point[0], goal_point[1], 'bx')


    if filename is not None:
        fig.savefig(filename)
    plt.show()


def get_random_valid_vertex(state_valid: Callable[[np.ndarray], bool], bounds: np.ndarray, obstacles: list) -> np.ndarray:
    # NOTE: Do not use this function in your implementation directly
    vertex = None
    while vertex is None: # Get starting vertex
        pt = np.random.rand(bounds.shape[0]) * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
        if state_valid(pt):
            vertex = pt
    return vertex


#####################################

K=250
def test_static_rrt_no_goal(rrts, steer, filenum):
    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = None
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    for version, rrt in enumerate(rrts):
        nodes = rrt(bounds, validity_check, starting_point, None, steer, K, np.linalg.norm(bounds/10.))
        visualize_2D_graph(bounds, obstacles, nodes, None, f'rrt{filenum}_static_{version}.png')

def test_random_rrt(rrts, steer, filenum):
    bounds, obstacles, validity_check = setup_random_2d_world()
    starting_point = None
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    for version, rrt in enumerate(rrts):
        nodes = rrt(bounds, validity_check, starting_point, None, steer, K, np.linalg.norm(bounds/10.))
        visualize_2D_graph(bounds, obstacles, nodes, None, f'rrt{filenum}_random_{version}.png')

def test_static_rrt_goal(rrts, steer, filenum):
    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    for version, rrt in enumerate(rrts):
        nodes = rrt(bounds, validity_check, starting_point, goal_point, steer, K, np.linalg.norm(bounds/10.))
        visualize_2D_graph(bounds, obstacles, nodes, goal_point, f'rrt{filenum}_goal_static_{version}.png')

def test_random_rrt_goal(rrts, steer, filenum):
    bounds, obstacles, validity_check = setup_random_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    for version, rrt in enumerate(rrts):
        nodes = rrt(bounds, validity_check, starting_point, goal_point, steer, K, np.linalg.norm(bounds/10.), obstacles)
        visualize_2D_graph(bounds, obstacles, nodes, goal_point, f'rrt{filenum}_goal_random_{version}.png')

def test_non_holonomic_rrt(rrts, steer, filenum):
    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    starting_point = np.append(starting_point,0.)
    bounds = np.append(bounds,[[-3.14159,3.14159]],0)
    for version, rrt in enumerate(rrts):
        nodes = rrt(bounds, validity_check, starting_point, None, steer, K*5, np.linalg.norm(bounds/10.))
        visualize_2D_graph(bounds, obstacles, nodes, None, f'rrt{filenum}_goal_static_{version}.png')

def test_non_holonomic_rrt_goal(rrts, steer, filenum):
    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    starting_point = np.append(starting_point,0.)
    goal_point = np.append(goal_point,0.)
    bounds = np.append(bounds,[[-3.14159,3.14159]],0)
    for version, rrt in enumerate(rrts):
        nodes = rrt(bounds, validity_check, starting_point, goal_point, steer, K*5, np.linalg.norm(bounds/10.))
        visualize_2D_graph(bounds, obstacles, nodes, goal_point, f'rrt{filenum}_goal_static_{version}.png')


if __name__ == "__main__":
    print("This file should not be run directly, please use the Jupyter Notebook instead.")
    pass