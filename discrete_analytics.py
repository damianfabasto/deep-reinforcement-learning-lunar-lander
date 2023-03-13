import pandas as pd
import gym
import numpy as np
import gym
from rlgtsrc.project2.deepqmodels import net
from rlgtsrc.project2.utils import ConfigParams
import rlgtsrc.project2.utils as utils
from tqdm.notebook import tqdm
import pdb

from collections import deque
from copy import deepcopy
import random

from abc import ABC, abstractmethod
import json
from pprint import pprint as pp
import os
import time




class DicreteMatrix_lunar_landing():

    def __init__(self, coordinateGrids:list, action_dim: int, params: ConfigParams):
        """

        :param coordinateGrids: [[-1.0, 0., 1.0], [0.0, 0.1, 0.5]]
        :param action_dim:
        """

        ignore_legs_touching_ground = params.dict["ignore_legs_touching_ground"]

        self.coordinateGrids = coordinateGrids

        # Initialize the Q matrix:
        dimensions = []
        for idx in range(len(coordinateGrids)):
            num_dim = len(self.coordinateGrids[idx]) + 2  # Add two to account of values below and above the given range
            dimensions.append(num_dim)

        if not ignore_legs_touching_ground:
            for _ in range(2):
                self.coordinateGrids.append([0, 1])
            for idx in range(2):
                dimensions.append(2)



        # Finally, add the action space
        dimensions.append(action_dim)
        self.coordinateGrids.append([idx for idx in range(action_dim)])


        # Initialize Q matrix
        self.Q = np.zeros(dimensions)

        self.ignore_legs_touching_ground = ignore_legs_touching_ground
        self.dimensions = dimensions
        self.N = len(dimensions)

    def _getQValueFromIndices(self, state_indices):
        # Is there a better way to do this?
        if self.ignore_legs_touching_ground:
            return self.Q[state_indices[0], state_indices[1], state_indices[2], state_indices[3], state_indices[4], state_indices[5], state_indices[6]]
        else:
            return self.Q[
                state_indices[0], state_indices[1], state_indices[2], state_indices[3], state_indices[4], state_indices[
                    5], state_indices[6], state_indices[7], state_indices[7]]

    def _setQValueFromIndices(self, state_indices, q_value):
        # Is there a better way to do this?
        if self.ignore_legs_touching_ground:
            self.Q[state_indices[0], state_indices[1], state_indices[2], state_indices[3], state_indices[4], state_indices[5]] = q_value
        else:
            self.Q[
                state_indices[0], state_indices[1], state_indices[2], state_indices[3], state_indices[4], state_indices[
                    5], state_indices[6], state_indices[7]] = q_value

    def getQValue(self, state_space, action):
        pdb.set_trace()
        state_space_ = np.array(state_space)
        if self.ignore_legs_touching_ground:
            state_space_ = state_space_[:-2]
            # Discretize state space
            state_indices = [np.digitize(state_space_[idx], self.coordinateGrids[idx]) for idx in range(self.N-1)]
        else:
            state_indices = [np.digitize(state_space_[idx], self.coordinateGrids[idx]) for idx in range(self.N - 3)]
            state_indices.extend(state_space_[-2:].astype(int))


        # Add the action
        state_indices.append(np.digitize(action, self.coordinateGrids[-1]))
        return self._getQValueFromIndices(state_indices)

    def greedyAction(self, current_state):
        actions = [0, 1, 2, 3]
        q_vals = [self.getQValue(current_state, action) for action in actions]
        return np.argmax(q_vals)

    def getMaxVal(self, current_state):
        greedyAction = self.greedyAction(current_state)
        q_val_max_a  = self.getQValue(current_state, greedyAction)
        return q_val_max_a

    def setQValue(self, state_space, action, q_value):
        state_space_ = np.array(state_space)
        if self.ignore_legs_touching_ground:
            state_space_ = state_space_[:-2]
            # Discretize state space
            state_indices = [np.digitize(state_space_[idx], self.coordinateGrids[idx]) for idx in range(self.N-1)]
        else:
            state_indices = [np.digitize(state_space_[idx], self.coordinateGrids[idx]) for idx in range(self.N - 3)]
            state_indices.extend(state_space_[-2:].astype(int))

        # Add the action
        state_indices.append(np.digitize(action, self.coordinateGrids[-1]))
        self._setQValueFromIndices(state_indices, q_value)


class DicreteMatrix_carpole():

    def __init__(self, coordinateGrids:list, action_dim: int, params: ConfigParams):
        """

        :param coordinateGrids: [[-1.0, 0., 1.0], [0.0, 0.1, 0.5]]
        :param action_dim:
        """



        self.coordinateGrids = coordinateGrids

        # Initialize the Q matrix:
        dimensions = []
        for idx in range(len(coordinateGrids)):
            num_dim = len(self.coordinateGrids[idx]) + 2  # Add two to account of values below and above the given range
            dimensions.append(num_dim)



        # Finally, add the action space
        dimensions.append(action_dim)
        self.coordinateGrids.append([idx for idx in range(action_dim)])


        # Initialize Q matrix
        self.Q = np.zeros(dimensions)


        self.dimensions = dimensions
        self.N = len(dimensions)

        self.actions = list(range(action_dim))

    def _getQValueFromIndices(self, state_indices):

        # Is there a better way to do this?
        return self.Q[state_indices[0], state_indices[1], state_indices[2], state_indices[3], state_indices[4]]

    def _setQValueFromIndices(self, state_indices, q_value):
        # Is there a better way to do this?
        self.Q[state_indices[0], state_indices[1], state_indices[2], state_indices[3], state_indices[4]] = q_value

    def getQValue(self, state_space, action):

        state_space_ = np.array(state_space)
        state_indices = [np.digitize(state_space_[idx], self.coordinateGrids[idx]) for idx in range(self.N - 1)]
        # Add the action
        state_indices.append(action)
        return self._getQValueFromIndices(state_indices)

    def greedyAction(self, current_state):
        q_vals = [self.getQValue(current_state, action) for action in self.actions]
        return np.argmax(q_vals)

    def getMaxVal(self, current_state):
        greedyAction = self.greedyAction(current_state)
        q_val_max_a  = self.getQValue(current_state, greedyAction)
        return q_val_max_a

    def setQValue(self, state_space, action, q_value):
        state_space_ = np.array(state_space)
        state_indices = [np.digitize(state_space_[idx], self.coordinateGrids[idx]) for idx in range(self.N - 1)]
        # Add the action
        state_indices.append(action)
        self._setQValueFromIndices(state_indices, q_value)


class Q_Based_Learning(ABC):

    def __init__(self, env, coordinateGrids:list, params: ConfigParams):

        self.env = env

        if params.dict["rl_problem"]=="carpole":
            self.Q = DicreteMatrix_carpole(coordinateGrids, env.action_space.n, params)
        else:
            self.Q = DicreteMatrix_lunar_landing(coordinateGrids, env.action_space.n, params)

        #self.actions_dict = {0: "do nothing", 1: "fire left engine", 2: "fire main engine", 3: "fire right engine"}
        #self.actions      = [0,1,2,3]

        self.actions_dict = {0: "push left", 1: "push right"}
        self.actions      = list(range(env.action_space.n))

        # Initialize the seed
        np.random.seed(params.dict["RANDOM_SEED"])

        self.params = params

    def providePolicy(self):
        optimalPolicy = np.argmax(self.Q, axis = -1)
        optimalPolicy = optimalPolicy.astype(int)
        optimalActions = []
        for action in optimalPolicy:
            optimalActions.append(self.actions_dict[action])
        return optimalActions

    @abstractmethod
    def learningUpdate(self, reward, current_state, current_action, next_sate, next_action, step_count):
        pass


    @abstractmethod
    def reset(self):
        """
        Implements any cleanin up or initialization that the algorithm may need
        :return:
        """
        pass

    def greedy(self, currrent_state):
        greedyAction = self.Q.greedyAction(currrent_state)
        return greedyAction

    def performEpsilonGreedy(self, currrent_state):
        # choose an action based on epsilon-greedy algorithm
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.greedy(currrent_state)

        return action


    def runOneEpisode(self):

        # Restart the environment and get starting state
        current_state = self.env.reset()
        # Restart the steps
        self.time = 1
        # Perform any reset, clearnup or initilization, that the learning algorithm may need
        self.reset()

        # Save actions taken
        actions_followed = []

        # Save the rewards
        total_rewards = 0

        reached_terminalSate = False



        # Choose action based on epsilon greedy
        current_action = self.performEpsilonGreedy(current_state)

        actions_followed.append(current_action)

        while not reached_terminalSate:
            # Perform action in the given state
            next_state, reward, reached_terminalSate, _  = self.env.step(int(current_action))

            # Get the next action
            next_action = self.performEpsilonGreedy(current_state)

            # Perform update
            self.learningUpdate(reward, current_state, current_action, next_state, next_action, self.time)

            # Update current_state and current_action
            current_state = next_state
            current_action = next_action

            actions_followed.append(current_action)

            self.time +=1

            total_rewards += reward



        return total_rewards, self.time, actions_followed

    def runEpisodes(self):

        total_rewards = []
        for episode_idx in tqdm(range(self.params.dict["nEpisodes"])):
            reward, steps, actions = self.runOneEpisode()
            total_rewards.append(reward)
        return total_rewards





class SarsaOnPolicy(Q_Based_Learning):

    def __init__(self, env, coordinateGrids:list, params: ConfigParams):

        super(__class__, self).__init__(env, coordinateGrids, params)


        self.epsilon = params.dict['epsilon']

        # Count the number of steps in episode
        self.time = 0



        self.alpha = params.dict['alpha']



    def learningUpdate(self, reward, current_state, current_action, next_sate, next_action, step_count):
        if step_count == self.env._max_episode_steps:
            Q_Sprime_Aprime = self.Q.getQValue(next_sate, next_action)
        else:
            Q_Sprime_Aprime = 0

        Q_S_A = self.Q.getQValue(current_state, current_action)

        target_value = self.alpha * (reward +  Q_Sprime_Aprime - Q_S_A)
        updated_value = Q_S_A + target_value

        # Sarsa update
        self.Q.setQValue(current_state, current_action, updated_value)



    def updateEpsilon(self, newEpsilon):
        self.epsilon = newEpsilon

    def reset(self):
        """
        This algorithm does not need to do any cleanup or initialization for each episode
        :return:
        """
        return






class QLearning(Q_Based_Learning):

    def __init__(self, env, coordinateGrids:list, params: ConfigParams):

        super(__class__, self).__init__(env, coordinateGrids, params)



        self.epsilon = params.dict["epsilon"]

        # Count the number of steps in episode
        self.time = 0

        self.alpha = params.dict["alpha"]

        self.gamma = params.dict["gamma"]

    def learningUpdate(self, reward, current_state, current_action, next_sate, next_action, step_count):
        if step_count == self.env._max_episode_steps:
            Q_Sprime_Max_A = self.Q.getMaxVal(current_state)
        else:
            Q_Sprime_Max_A = 0

        Q_S_A = self.Q.getQValue(current_state, current_action)

        target_value = self.alpha * (reward + Q_Sprime_Max_A - Q_S_A)
        updated_value = Q_S_A + target_value

        # Sarsa update
        self.Q.setQValue(current_state, current_action, updated_value)


    def updateEpsilon(self, newEpsilon):
        self.epsilon = newEpsilon


    def reset(self):
        """
        This algorithm does not need to do any cleanup or initialization for each episode
        :return:
        """
        return



class ExpectedQLearning(Q_Based_Learning):

    def __init__(self, env, coordinateGrids:list, params: ConfigParams, policy):

        super(__class__, self).__init__(env, coordinateGrids, params)

        self.epsilon = params.dict["epsilon"]

        # Count the number of steps in episode
        self.time = 0

        self.alpha = params.dict["alpha"]

        self.gamma = params.dict["gamma"]

        self.policy = policy



    def learningUpdate(self, reward, current_state, current_action, next_sate, next_action, step_count):

        if step_count == self.env._max_episode_steps:
            Q_Sprime_Max_A = np.sum([self.policy[a] * self.Q.getQValue(next_sate, a) for a in self.actions])
        else:
            Q_Sprime_Max_A = 0

        Q_S_A = self.Q.getQValue(current_state, current_action)

        target_value = self.alpha * (reward + Q_Sprime_Max_A - Q_S_A)
        updated_value = Q_S_A + target_value

        # Sarsa update
        self.Q.setQValue(current_state, current_action, updated_value)


    def updateEpsilon(self, newEpsilon):
        self.epsilon = newEpsilon


    def reset(self):
        """
        This algorithm does not need to do any cleanup or initialization for each episode
        :return:
        """
        return



class SarsaLambda(Q_Based_Learning):

    def __init__(self, env, coordinateGrids:list, ignore_legs_touching_ground, seed,
                 epsilon, lbda, alpha, gamma):

        super(__class__, self).__init__(env, coordinateGrids, ignore_legs_touching_ground, seed)

        self.epsilon = epsilon

        # Count the number of steps in episode
        self.time = 0

        self.alpha = alpha

        self.gamma = gamma

        self.lbda = lbda

        # Initialize the eligibility traces
        self.reset()

    def reset(self):
        """
        This algorithm requires to initialize the eligibility traces to zero at the begining of each episode!
        """
        # state x actions
        self.eligTraces = self.DicreteMatrix(self.coordinateGrids, self.env.action_space.n, self.ignore_legs_touching_ground)
        return

    def learningUpdate(self, reward, current_state, current_action, next_sate, next_action, step_count):

        if step_count == self.env._max_episode_steps:
            Q_Sprime_Aprime = self.Q.getQValue(next_sate, next_action)
        else:
            Q_Sprime_Aprime = 0

        delta = reward + self.gamma * Q_Sprime_Aprime - self.Q.getQValue(current_state, current_action)


        # Update eligibility
        updated_eligibility = self.eligTraces.getQValue(current_state, current_action) + 1
        self.eligTraces.setQValue(current_state, current_action, updated_eligibility)

        # Now loop over all the actions and states
        self.Q.Q += (self.alpha * delta * self.eligTraces.Q)
        self.eligTraces.Q *= self.gamma * self.lbda * self.eligTraces.Q














