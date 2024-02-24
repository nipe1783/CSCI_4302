from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import hw2_rrt
from hw2_rrt import Node, visualize_2D_graph

def get_nearest_vertex(node_list: list[Node], q_point: np.ndarray) -> Node:
    '''
    @param node_list: List of Node objects
    @param q_point: Query point, a numpy array of appropriate dimension
    @return Node in node_list with the closest node.point to the query q_point
    '''
    min_dist = float('inf')
    nearest_node = None
    for node in node_list:
        dist = np.linalg.norm(node.point - q_point)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node

def steer_holonomic(from_point: np.ndarray, to_point: np.ndarray, delta_q: float) -> np.ndarray:
    '''
    @param from_point: Point where the path to "to_point" is originating from
    @param to_point: Point indicating destination
    @param delta_q: Max path-length to cover
    @returns path: list of points leading from "from_point" to "to_point" (inclusive of endpoints)
    '''
    # Calculate the direction vector from from_point to to_point
    direction_vector = to_point - from_point    
    # Normalize the direction vector to get a unit vector
    unit_direction_vector = direction_vector / np.linalg.norm(direction_vector)
    # Scale the unit vector by delta_q to get a vector of length delta_q
    delta_vector = unit_direction_vector * delta_q
    # Compute the new point by adding the delta_vector to from_point
    new_point = from_point + delta_vector
    return new_point

# Use this area to define any helper functions you'd like to use
def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    '''
    @param point1: Numpy array of appropriate dimension
    @param point2: Numpy array of appropriate dimension
    @return Euclidean distance between point1 and point2
    '''
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i])**2
    return math.sqrt(distance)

def random_point(state_bounds: np.ndarray) -> np.ndarray:
    '''
    @param state_bounds: Numpy array of appropriate dimension
    @return Random point within the bounds given by state_bounds
    '''
    dimensions = len(state_bounds)
    point = np.zeros(dimensions)
    for i in range(dimensions):
        point[i] = random.uniform(state_bounds[i][0], state_bounds[i][1])
    return point

def generate_path(from_point: np.ndarray, to_point: np.ndarray, delta_q: float) -> list[np.ndarray]:
    '''
    @param from_point: Point where the path to "to_point" is originating from
    @param to_point: Point indicating destination
    @param delta_q: Max path-length to cover
    @returns path: list of points leading from "from_point" to "to_point" (inclusive of endpoints)
    '''
    steps = 100
    step_size = delta_q / steps
    path = [from_point]
    # Calculate the direction vector from from_point to to_point
    direction_vector = to_point - from_point    
    # Normalize the direction vector to get a unit vector
    unit_direction_vector = direction_vector / np.linalg.norm(direction_vector)
    for i in range(1, steps):
        delta_vector = unit_direction_vector * step_size * i
        path.append(from_point + delta_vector)
    path.append(to_point)
    return path

def path_is_valid(from_point: np.ndarray, to_point: np.ndarray, delta_q: float, state_is_valid: Callable[[np.ndarray],bool]) -> bool:

    path = generate_path(from_point, to_point, delta_q)
    for point in path:
        if not state_is_valid(point):
            return False
    return True

def get_nearest_nodes(node_list: list[Node], q_point: np.ndarray, delta_q: float) -> list[Node]:
    
    nearest_nodes = []
    for node in node_list:
        if distance(node.point, q_point) < delta_q:
            nearest_nodes.append(node)
    return nearest_nodes

def update_path_from_parent(node):

    if node.parent:
        node.path_from_parent = np.vstack((node.parent.path_from_parent, node.point))
        node.path_length = node.parent.path_length + distance(node.point, node.parent.point)
    for child in node.children:
        update_path_from_parent(child)

def rrt_star(state_bounds, state_is_valid, starting_point, goal_point, steer, k, delta_q, obstacles):
    node_list = []
    root_node = Node(starting_point)
    root_node.path_from_parent = np.array([starting_point])
    root_node.path_length = 0
    node_list.append(root_node)
    goal_bias = 0.1
    goal_path_length = float('inf')
    goal_parent = None

    for i in range(k):
        # Bias towards the goal with a probability of goal_bias
        if np.random.rand() < goal_bias and goal_point is not None:
            q_rand = goal_point
        else:
            q_rand = random_point(state_bounds)

        nearest_node = get_nearest_vertex(node_list, q_rand)
        q_new = steer(nearest_node.point, q_rand, delta_q)
        neighbors = get_nearest_nodes(node_list, q_new, delta_q)
        min_cost = float('inf')
        min_node = None

        # Find the best parent for the new node
        for neighbor in neighbors:
            if path_is_valid(neighbor.point, q_new, delta_q, state_is_valid):
                cost = neighbor.path_length + np.linalg.norm(neighbor.point - q_new)
                if cost < min_cost:
                    min_cost = cost
                    min_node = neighbor

        # If a valid parent is found, create and add the new node
        if min_node:
            new_node = Node(q_new, min_node)
            new_node.path_from_parent = np.array([min_node.point, q_new])
            new_node.path_length = min_cost
            min_node.children.append(new_node)  # Add the new node to the parent's children list
            node_list.append(new_node)

            # Reconnect neighbors if a better path is found through the new node
            for neighbor in neighbors:
                if neighbor == min_node:
                    continue  # Skip the parent node
                potential_cost = new_node.path_length + np.linalg.norm(q_new - neighbor.point)
                if potential_cost < neighbor.path_length and path_is_valid(q_new, neighbor.point, delta_q, state_is_valid):
                    # Update the neighbor's parent if a better path is found
                    if neighbor.parent:
                        neighbor.parent.children.remove(neighbor)  # Remove from old parent's children list
                    neighbor.parent = new_node
                    neighbor.path_from_parent = np.array([q_new, neighbor.point])
                    neighbor.path_length = potential_cost
                    new_node.children.append(neighbor)  # Add to new parent's children list

            # Check if this node creates a new best path to the goal
            if goal_point is not None and np.linalg.norm(q_new - goal_point) < delta_q and path_is_valid(q_new, goal_point, delta_q, state_is_valid) and new_node.path_length + np.linalg.norm(q_new - goal_point) < goal_path_length:
                goal_parent = new_node
                goal_path_length = new_node.path_length + np.linalg.norm(q_new - goal_point)

    # Create the goal node and add it to the node list if a path to the goal was found
    if goal_parent:
        goal_node = Node(goal_point, goal_parent)
        goal_node.path_from_parent = np.array([goal_parent.point, goal_point])
        goal_node.path_length = goal_path_length
        goal_parent.children.append(goal_node)  # Add the goal node to the parent's children list
        node_list.append(goal_node)

    return node_list


# hw2_rrt.test_static_rrt_no_goal([rrt, rrt_star], steer_holonomic, '01')
hw2_rrt.test_random_rrt_goal([rrt_star, rrt_star], steer_holonomic, '06')