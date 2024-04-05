# CSCI 4302/5302 HW2
# (C) Brad Hayes <bradley.hayes@colorado.edu> 2023
version = "v2023.10.5.0000"

import gym
import gym_mountaincar
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


student_name = "Nicolas Perrault" # Set to your name
GRAD = False  # Set to True if graduate student


class TabularPolicy(object):
    def __init__(self, n_bins_per_dim, num_dims, n_actions):
        self.num_states = n_bins_per_dim ** num_dims        
        self.num_actions = n_actions
        
        self._transition_function = np.zeros(shape=(self.num_states, self.num_actions, self.num_states))
        self._reward_function = np.zeros(shape=(self.num_states, self.num_actions, self.num_states))

        # Create data structure to store mapping from state to value
        # self._value_function[state] = state value
        self._value_function = np.zeros(shape=(self.num_states,))

        # Create data structure to store array with probability of each action for each state
        # self._policy[state] = [array of action probabilities]
        self._policy = np.random.uniform(0, 1, size=(self.num_states, self.num_actions))       

    def get_action(self, state):
        '''
        Returns an action drawn from the policy's action distribution at state
        '''
        prob_dist = np.array(self._policy[state])
        assert prob_dist.ndim == 1

        # Sample from policy distribution for state
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html
        idx = np.random.multinomial(1, prob_dist / np.sum(prob_dist))
        return np.argmax(idx)

    def set_state_value(self, state, value):
        '''
        Sets the value of a given state
        '''
        self._value_function[state] = value
    
    def get_state_value(self, state):
        '''
        Returns the value of a given state
        '''
        return self._value_function[state]

    def get_value_function(self):
        '''
        Returns the table representing the value function itself.
        Useful to do for storing a value function at the end of an iteration of VI or PI...
        '''
        return copy.deepcopy(self._value_function)

    def set_value_function(self, v):
        '''
        Sets value function with new matrix v
        '''
        self._value_function = copy.copy(v)

    def set_policy(self, state, action_prob_array):
        '''
        Sets the action probabilities for a given state
        '''
        self._policy[state] = copy.copy(action_prob_array)

    def get_policy(self, state):
        '''
        Returns the probability distribution over actions for a given state
        '''
        return self._policy[state]

    def get_policy_function(self):
        '''
        Returns the table representing the policy function itself.
        '''
        return copy.deepcopy(self._policy)

    def set_entire_policy_function(self, p):
        '''
        Sets value function with new matrix v
        '''
        self._policy = copy.deepcopy(p)

    def set_entire_transition_function(self, t):
        '''
        Sets the entire transition function
        '''
        self._transition_function = copy.deepcopy(t)

    def set_transition_function(self, state_idx, action_idx, next_state_idx, val):
        '''
        Sets the transition function entry for a (s,a,s') tuple.
        '''
        self._transition_function[state_idx, action_idx, next_state_idx] = val

    def T(self, state_idx, action_idx, next_state_idx):
        '''
        Gets the transition function entry for a (s,a,s') tuple.
        '''
        return self._transition_function[state_idx, action_idx, next_state_idx]

    def set_reward_function(self, state_idx, action_idx, next_state_idx, val):
        '''
        Sets the reward function entry for a (s,a,s') tuple.
        '''
        self._reward_function[state_idx, action_idx, next_state_idx] = val

    def R(self, state_idx, action_idx, next_state_idx):
        '''
        Gets the reward function entry for a (s,a,s') tuple.
        '''
        return self._reward_function[state_idx, action_idx, next_state_idx]

class DiscretizedSolver(object):
    def __init__(self, mode, num_bins=21, lookahead=-1):
        self._mode = mode
        self._lookahead_steps = lookahead
        assert mode in ['nn', 'linear', 'lookahead']
        self._num_bins = num_bins

        self.env = gym.make("mountaincar5302-v0") # Problem Environment
        self.env_name = 'MountainCar'
        start_state = self.env.reset()

        self.state_lower_bound = self.env.observation_space.low
        self.state_upper_bound = self.env.observation_space.high
        self.bin_sizes = (self.state_upper_bound - self.state_lower_bound) / self._num_bins
        self.num_dims = self.state_lower_bound.shape[0]
        self.gamma = 0.99
        self.solver = TabularPolicy(self._num_bins, self.num_dims, self.env.action_space.n)
        self.performance_history = [] # List of cumulative rewards from successive runs of the policy
        self.populate_transition_and_reward_funcs() # Fill in our own T and R functions by using the simulator for the environment!


    def populate_transition_and_reward_funcs(self):
        '''
        Making use of self.add_transition, this function sets up the last of our discretized MDP variables
        T and R by iterating over all of our discrete states and filling in the appropriate values.
        '''

        # Iterate over all state/action combinations
        # Call self.add_transition for each
        for state_idx in range(self.solver.num_states):
            for action_idx in range(self.env.action_space.n):
                self.add_transition(state_idx, action_idx)
        

    def add_transition(self, state_idx, action_idx):
        '''
        Sets the discretized MDP transition and reward values for a given state-action pair.
        state_idx: int corresponding to the 'from' state
        action_idx: int corresponding to the action taken
        '''

        # MountainCar is deterministic, so you do not need to sample multiple times for each state/action pair
        # in order to find out T and R.

        # HINT: You can set self.env.state to the state you wish to simulate (e.g., self.env.state = [0.001, 0.5])
        # HINT: You can use the self.env.step function to simulate an action. It returns 4 values: next_state, reward, environment_complete, info.
        # HINT: Keep in mind that we're setting T and R for our approximate, discretized MDP! You'll need to map the continuous state onto your discrete state space somehow.

        # Your code here. Remember to set_transition_function and set_reward_function on self.solver to store the values you compute!
        # Get the continuous coordinates corresponding to the center of the current discrete state
        self.env.state = self.get_coordinates_from_state_index(state_idx)
        next_state, reward, done, info = self.env.step(action_idx)
        next_state_idx = self.get_state_index_from_coordinates(next_state)
        next_state = self.get_coordinates_from_state_index(next_state_idx)
        self.solver.set_transition_function(state_idx, action_idx, next_state_idx, 1)
        self.solver.set_reward_function(state_idx, action_idx, next_state_idx, reward)

    def get_discrete_state_probabilities(self, continuous_state_vector):
        '''
        Given a continuous state (e.g., [1.2,-0.3]), 
        return a list of (discrete state index, probability of being in this state) tuples

        For mode=='nn' (nearest neighbor), this list will only have one element, consisting of the **state index**
        closest to continuous_state_vector e.g.,: [(6, 1.0)] to indicate state index 6 with 100% probability.

        For mode=='linear', this list will have multiple elements, indicating the nearest discrete states.
        Specifically, it should have 2^(state_dimension) elements, each containing (state index, probability). Mountain Car is a 2D problem.
        
        To build intuition, for a 1-dimensional problem this would indicate interpolation between two states (e.g., for a 1D continuous state space with 3 discrete states S1 S2 S3, given continuous state "Sx":  S1-----Sx---S2----------S3, it would return a list containing state indices 1 and 2 along with the probability of each, but not state 3.)
        '''
        # HINT: Computing distances to every state in your discrete state space is going to be slow! Try to come up with a faster solution.

        if self._mode == 'nn':
            # Find the closest discrete state to continuous_state_vector
            discrete_state_index = self.get_state_index_from_coordinates(continuous_state_vector)
            return [(discrete_state_index, 1.0)]
        elif self._mode == 'linear':
            # Find the 4 discrete states that surround continuous_state_vector, and assign
            # probability inversely proportionate to their distance from it.
            # Calculate the position of the continuous state relative to the bin sizes
            state_position = (continuous_state_vector - self.state_lower_bound) / self.bin_sizes
            
            # Find the indices of the lower bounds of the bin that the state falls into
            lower_bin_indices = np.floor(state_position).astype(int)
            
            # Calculate the distances of the state from the lower bounds
            distances = state_position - lower_bin_indices
            
            # The weights should be proportional to the inverse of the distance
            weights = 1 - distances
            
            # Get the index of the lower bound state
            lower_state_idx = self.get_state_index_from_coordinates(self.state_lower_bound + lower_bin_indices * self.bin_sizes)
            
            # Compute the indices of the four corners and their weights
            state_probabilities = []
            for i in range(2):
                for j in range(2):
                    neighbor_bin_indices = lower_bin_indices.copy()
                    neighbor_bin_indices[0] += i
                    neighbor_bin_indices[1] += j
                    weight = weights[0] if i == 0 else (1 - weights[0])
                    weight *= weights[1] if j == 0 else (1 - weights[1])
                    neighbor_state_idx = self.get_state_index_from_coordinates(self.state_lower_bound + neighbor_bin_indices * self.bin_sizes)
                    state_probabilities.append((neighbor_state_idx, weight))
                    
            return state_probabilities

    def compute_policy(self, max_iterations=150):
        '''
        Compute a policy and store it in self.solver (type TabularPolicy) using Value Iteration for max_iterations iterations.
        '''
        # HINT: Append the cumulative reward from self.solve() into self.performance_history each iteration
        #       Make sure to pick a reasonable value for solve()'s max_steps parameter (i.e., less than infinity)

        # Your Code Here -- You should be able to reuse most of your VI code from tabular_solution.py here

        # GRADs: Make sure to include code supporting the policy evaluation for n-step lookahead at the end of each iteration.
        #        You can tell if you're using n-step lookahead because the self._lookahead_steps variable will be set
        #        You can use self.solver.set_policy_function and self.solve to get reward values to append to self.performance_history

        # HINT: Because this one can take a while to converge, set up a stopping criteria.
        #       If ||(v_i+1 - v_i)||_2 is small (i.e., distance between v_i+1 and v_i is smaller than 1e-3), the value function isn't updating much and you can safely stop iterating.


        # Remember to call self.solver.set_policy/value_function at the end!
        print("MODE:" , self._mode)
        V = np.zeros(self.solver.num_states)
        V_prev = V.copy()
        policy = np.zeros(self.solver.num_states, dtype=int)
        theta = 1e-6
        delta = 0
        for iteration in range(max_iterations):
            delta_prev = delta
            delta = 0
            for s in range(self.solver.num_states):
                v = V[s]
                action_values = np.zeros(self.env.action_space.n)
                for a in range(self.env.action_space.n):
                    self.env.reset()
                    continuous_state = self.get_coordinates_from_state_index(s)
                    self.env.state = continuous_state
                    next_state, reward, done, _ = self.env.step(a)
                    # next_state_idx = self.get_state_index_from_coordinates(next_state)
                    state_probabilities = self.get_discrete_state_probabilities(next_state)
                    for state_idx, prob in state_probabilities:
                        action_values[a] += prob * (reward + self.gamma * V[state_idx])
                best_action_value = np.max(action_values)
                policy[s] = np.argmax(action_values)
                delta = max(delta, np.abs(v - best_action_value))
                V[s] = best_action_value
            print("change in v:", delta_prev - delta)
            if delta_prev - delta < theta and delta_prev - delta >= 0:
                break

            
            for s in range(self.solver.num_states):
                action_values = np.zeros(self.env.action_space.n)
                for a in range(self.env.action_space.n):
                    self.env.reset()
                    continuous_state = self.get_coordinates_from_state_index(s)
                    self.env.state = continuous_state
                    next_state, reward, done, _ = self.env.step(a)
                    state_probabilities = self.get_discrete_state_probabilities(next_state)
                    for state_idx, prob in state_probabilities:
                        action_values[a] += prob * (reward + self.gamma * V[state_idx])
                best_action = np.argmax(action_values)
                self.solver.set_state_value(s, V[s])
                self.solver.set_policy(s, np.eye(self.env.action_space.n)[best_action])
            cumulative_reward, steps = self.solve(False, 500)
            self.performance_history.append(cumulative_reward)

        for s in range(self.solver.num_states):
            action_values = np.zeros(self.env.action_space.n)
            for a in range(self.env.action_space.n):
                self.env.reset()
                continuous_state = self.get_coordinates_from_state_index(s)
                self.env.state = continuous_state
                next_state, reward, done, _ = self.env.step(a)
                state_probabilities = self.get_discrete_state_probabilities(next_state)
                for state_idx, prob in state_probabilities:
                    action_values[a] += prob * (reward + self.gamma * V[state_idx])
            best_action = np.argmax(action_values)
            self.solver.set_state_value(s, V[s])
            self.solver.set_policy(s, np.eye(self.env.action_space.n)[best_action])
        self.solver.set_value_function(V)
        reward = self.solve( False, 150)
        print("reward: ", reward[0], "steps: ", reward[1])
        self.performance_history.append(reward)



    def get_state_index_from_coordinates(self, continuous_state_vector):
        # Compute bin indices for each dimension
        bin_indices = (continuous_state_vector - self.state_lower_bound) / self.bin_sizes
        bin_indices = np.floor(bin_indices).astype(int)
        bin_indices = np.clip(bin_indices, 0, self._num_bins - 1)
        state_index = bin_indices[0] * self._num_bins + bin_indices[1]
        return state_index

    def get_coordinates_from_state_index(self, state_idx):
        '''
        Returns the continuous state vector for a given discrete state index, returning
        the coordinates from the "middle" of discrete state cell
        '''
        bin_sizes = (self.state_upper_bound - self.state_lower_bound) / self._num_bins
        coordinates = np.array([(state_idx//self._num_bins + 0.5) * bin_sizes[0], (state_idx % self._num_bins + 0.5) * bin_sizes[1]]) + self.state_lower_bound
        return coordinates

    def evaluate_action_sequence(self, start_state, action_sequence):
        '''
        Simulate forward N steps from start_state using action_sequence, then return the accumulated reward
        ex: evaluate_action_sequence([0.4,0.2], [1,1,1])
        '''
        raise NotImplementedError

    def solve(self, visualize=False, max_steps=float('inf')):
        '''
        Reset the environment, then applies the solver's policy. 
        Returns cumulative reward and number of actions taken.
        '''

        finished = False
        cur_state = self.env.reset()
        if visualize is True: env.render()

        cumulative_reward = 0
        num_steps = 0

        while finished is False and num_steps < max_steps:
            # Take an action in the environment
            action = None

            if self._lookahead_steps > 0:
                # Your Code Here -- Implement n-step lookahead here for n = self._lookahead_steps
                # Generate all n-length action sequences and evaluate them using the evaluate_action_sequence function above
                # Remember to record the start state before simulating so you can reset it after you've found an action sequence to execute the first action of.
                # HINT: itertools.product can help you find all the action sequences.
                action = 0 # TODO: Replace this line!
                raise NotImplementedError
            elif self._mode == 'nn':
                discrete_state_probabilities = self.get_discrete_state_probabilities(cur_state)
                nearest_neighbor_state_index, _ = discrete_state_probabilities[0]
                action = self.solver.get_action(nearest_neighbor_state_index)
            elif self._mode == 'linear':
                state_probabilities = self.get_discrete_state_probabilities(cur_state)
                action_values = np.zeros(self.env.action_space.n)
                for state_idx, prob in state_probabilities:
                    action_probabilities = self.solver.get_policy(state_idx)
                    action_values += prob * action_probabilities
                action = np.argmax(action_values)
            else:
                action = self.solver.get_action(self.get_state_index_from_coordinates(cur_state))

            # Execute Action            
            next_state, reward, finished, info = self.env.step(action)
            
            # Update state
            cur_state = next_state
            
            # Update cumulative reward
            cumulative_reward += reward

            # Update action counter
            num_steps += 1

            # Display new state of the world
            if visualize is True: self.env.render()

        return cumulative_reward, num_steps

        
    def plot_value_function(self, value_function, filename=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        ax = fig.axes[0]

        V = ((value_function - value_function.min()) / (value_function.max() - value_function.min() + 1e-6))
        V = V.reshape(self._num_bins, self._num_bins).T
        image = (plt.cm.coolwarm(V)[::-1,:,:-1] * 255.).astype(np.uint8)
        ax.set_title("Env: %s" % self.env_name )
        ax.imshow(image)
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas.draw()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height),int(width),3)

        if filename is None:
            path = "./figures"
            if os.path.isdir(path) is False:
                if os.path.isdir("../figures") is True:
                    path = "../figures"
                else:
                    os.mkdir(path)
            filename = path+"/%s_%s_%d.png" % (self.env_name, self._mode, self._num_bins)

        fig.savefig(filename)

        return image, fig


def plot_policy_curves(reward_histories, filename=None):
    plt.close()
    plt.clf()
    symbols = ['bs', 'r--', 'g^']
    for idx, reward_history in enumerate(reward_histories):
        plt.plot(range(len(reward_history)), reward_history, symbols[idx%len(symbols)])

    plt.xlabel("Iteration")
    plt.ylabel("Return")
    plt.title("Policy Iteration Performance" )

    plt.savefig(filename)
    return


if __name__ == '__main__':
    ############ Q3.a ############
    bin_counts = [21, 51, 101]
    for bin_count in bin_counts:
        mc_solver = DiscretizedSolver(mode='nn', num_bins=bin_count, lookahead=-1)
        start_time = time.time()
        mc_solver.compute_policy()
        elapsed_time = time.time() - start_time
        print("Computed Q3.a VI Policy with bin size %d in %g seconds" % (bin_count, elapsed_time))
        mc_solver.plot_value_function(mc_solver.solver.get_value_function())

    ############ Q3.b ############
    linear_solvers = []
    bin_counts = [21, 51, 101]
    for bin_count in bin_counts:
        mc_solver = DiscretizedSolver(mode='linear', num_bins=bin_count, lookahead=-1)
        start_time = time.time()
        mc_solver.compute_policy()
        elapsed_time = time.time() - start_time
        print("Computed Q3.b VI Policy with bin size %d in %g seconds" % (bin_count, elapsed_time))
        mc_solver.plot_value_function(mc_solver.solver.get_value_function())
        linear_solvers.append(mc_solver)

    ############ Q4 ############
    if GRAD is True:
        n_step = [1,2,3]
        performance = []
        for lookahead in n_step:
            mc_solver = DiscretizedSolver(mode='linear', num_bins=51, lookahead=lookahead)
            start_time = time.time()
            mc_solver.compute_policy()
            elapsed_time = time.time() - start_time
            print("Computed Q4 Lookahead Policy in %g seconds" % elapsed_time)
            performance.append(mc_solver.performance_history)
        plot_policy_curves(performance, "figures/lookahead_mc.png")

