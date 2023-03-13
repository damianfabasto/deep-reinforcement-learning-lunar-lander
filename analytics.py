import pandas as pd
import gym
import numpy as np
import gym
from rlgtsrc.project2.deepqmodels import net
from rlgtsrc.project2.utils import ConfigParams
import rlgtsrc.project2.utils as utils
from tqdm.notebook import tqdm
import pdb
import torch
from collections import deque
from copy import deepcopy
import random

from abc import ABC, abstractmethod
import json
from pprint import pprint as pp
import os
import time
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import pickle


def simplePolicyCartPole(obs):
    """
    Simple policy, whereby we push the cart to the left when the linear velocity is to the right, and viceversa
    :param obs:
    :return:
    """
    angular_velocity = obs[-1] # positive means clockwise
    push_cart_to_the_left  = 0
    push_cart_to_the_right = 1
    return push_cart_to_the_right if angular_velocity > 0 else push_cart_to_the_left


def simplePolicyLunarLander(obs):
    """
    Simple policy, whereby we activate the left orientation engine if the craft has negative angular velocity, or the right
    orientation engine if it has positive angular velocity
    :param obs:
    :return:
    """
    angular_velocity = obs[5]
    return 1 if angular_velocity < 0 else 3


def runEpisodeSimplePolicy(env, n_episodes, n_steps, simplePolicy):
    totals = []
    frames_one_episode = []
    for episode in range(n_episodes):
        episode_rewards = 0
        obs = env.reset()
        for step in range(n_steps):

            if episode == 0:
                frames_one_episode.append(env.render(mode="rgb_array"))

            action = simplePolicy(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break
        totals.append(episode_rewards)
    # CLose the environment, otherwise the gym open an animation and freezes my Python.
    env.close()
    return totals, frames_one_episode


class Replay():

    def __init__(self, replay_size, samplingMethod = "random", device = torch.device("cpu")):

        #self.replay = []
        self.replay = deque(maxlen = replay_size)
        self.experience_size = None

        self.sample = lambda batchData: self._randomSample(batchData) if samplingMethod == "random" else self._serialSample(batchData)

        self.device = device

    def insert(self, experience: tuple):
        self.replay.append(experience)
        self.experience_size = len(experience)

    def unpackReplayData(self, batchData):
        # Unpack the data
        current_state, current_action, rewards, next_sate, episode_terminated = [
            np.array([obs[idx] for obs in batchData]) for idx in range(self.experience_size)]

        with torch.no_grad():
            return torch.from_numpy(current_state).float().to(self.device), \
                   torch.from_numpy(current_action).long().to(self.device), \
                   torch.from_numpy(rewards).float().unsqueeze(1).to(self.device), \
                   torch.from_numpy(next_sate).float().to(self.device), \
                   torch.from_numpy(episode_terminated).float().unsqueeze(1).to(self.device)


    def _randomSample(self, batchSize):
        rand_indices = [np.random.choice(range(len(self.replay))) for _ in range(batchSize)]
        batchData = [self.replay[idx] for idx in rand_indices]
        return self.unpackReplayData(batchData)

    def _serialSample(self, batchSize):
        # Draw in the same order they were inserted, starting from the last
        batchData = [self.replay[-idx] for idx in range(1, batchSize+1)]
        return self.unpackReplayData(batchData)


def checkParamsConsistency(params:ConfigParams):
    samplingMethod = params.dict["samplingMethod"]
    if not samplingMethod in ["serial", "random"]:
        raise ValueError("samplingMethod should be either 'serial' or 'random' - {} was given instead".format(samplingMethod))
    if samplingMethod == "serial":
        if params.dict["minEpisodesToTrainNet"] < params.dict["batchSize"] and params.dict["fillUpReplayBufferAtStart"]==False:
            raise ValueError("""minEpisodesToTrainNet should be at least as large as batchSize when the replay is serial. Otherwise, the replay won't have
                             "enough samples to pull from in the same order they were inserted""")


class EpsilonScheduler():

    def __init__(self, params:ConfigParams):

        self.params  = params
        self.rewards = deque(maxlen = params.dict["rewardWindowForEpsilonSchedule"])

        if self.params.dict["minNumEpisodesBeforeJumpToZero"] <= 1:
            self.episodesThresholdToJumpToZero = int(self.params.dict["minNumEpisodesBeforeJumpToZero"] * self.params.dict["nEpisodes"])
        else:
            self.episodesThresholdToJumpToZero = self.params.dict["minNumEpisodesBeforeJumpToZero"]



    def updateRewards(self, reward):
        self.rewards.append(reward)

    # Various different implementations of epsilon schedule
    def linearEpsilon(self, episode_idx):
        episode_factor = self.params.dict["nEpisodes"] // self.params.dict["numEpisodesToEndEpsilonRampDown"]
        return max(min(1.0, self.params.dict["minEpisodesFullExploration"] / (episode_factor * episode_idx + 1)),
                   self.params.dict["min_epsilon"])

    def linearEpsilonJumpToZero(self, episode_idx):
        if episode_idx > self.episodesThresholdToJumpToZero:
            return 0.0
        return self.linearEpsilon(episode_idx)

    def constantEpsilon(self):
        return self.params.dict["min_epsilon"]

    def constantJumpZeroEpsilon(self, episode_idx):
        if episode_idx > self.episodesThresholdToJumpToZero:
            return 0.0
        return self.params.dict["min_epsilon"]

    def constantScaleRampDownJumpZero(self, episode_idx):
        if episode_idx > self.episodesThresholdToJumpToZero:
            return 0.0
        return max(self.params.dict["epsilon_start"] * self.params.dict["epsilon_decay"]**episode_idx, self.params.dict["min_epsilon"])

    def constantScaleRampDown(self, episode_idx):
        return self.params.dict["epsilon_start"] * self.params.dict["epsilon_decay"]**episode_idx

    def constantScaleRampDownWithMin(self, episode_idx):
        return max(self.params.dict["min_epsilon"], self.params.dict["epsilon_start"] * self.params.dict["epsilon_decay"]**episode_idx)

    def hyperbolic(self, episode_idx):
        if episode_idx > self.episodesThresholdToJumpToZero:
            return 0.0
        return 1.0/(1.0 + episode_idx)

    def getEpsilon(self, episode_idx):

        epsilon_scheduler = self.params.dict["epsilon_scheduler"]
        if epsilon_scheduler == "ramp_down":
            return self.linearEpsilon(episode_idx)
        elif epsilon_scheduler == "ramp_down_jump_zero":
            return self.linearEpsilonJumpToZero(episode_idx)
        elif epsilon_scheduler == "constant":
            return self.constantEpsilon()
        elif epsilon_scheduler == "constant_scale_ramp_down":
            return self.constantScaleRampDown(episode_idx)
        elif epsilon_scheduler == "constant_scale_ramp_down_jump_zero":
            return self.constantScaleRampDownJumpZero(episode_idx)
        elif epsilon_scheduler == "hyperbolic":
            return self.hyperbolic(episode_idx)
        elif epsilon_scheduler == "constant_jump_zero":
            return self.constantJumpZeroEpsilon(episode_idx)
        elif epsilon_scheduler == "constantScaleRampDownWithMin":
            return self.constantScaleRampDownWithMin(episode_idx)



        else:
            raise ValueError("Epsilon schedule {} not recognized!".format(epsilon_scheduler))






class DeepQ(ABC):

    def __init__(self, env, params:ConfigParams, device = torch.device("cpu"), experimentName= None, envName = "LunarLander-v2"):

        # Sanity checks on the parameters
        checkParamsConsistency(params)

        self.setSeeds(seed = params.dict["RANDOM_SEED"])

        # How many episodes will it run with 100% exploration
        self.minEpisodesFullExploration           = params.dict["minEpisodesFullExploration"]

        self.params = deepcopy(params)

        self.epsilonScheduler = EpsilonScheduler(params)

        self.greedyScheme = self._greedy_softmax if params.dict["greedy_scheme"]=="softmax" else self._greedy

        self.device = device

        self.experimentName = experimentName

        self.env = env

        self.envName = envName

        self.numTestRuns = 0

        self.test_run_mean_rewards = []

    def saveModel(self, model, episode_idx, last_epsilon, optimizer, is_best):
        # Save weights
        utils.save_checkpoint({'epoch': episode_idx + 1,
                               'last_epsilon': last_epsilon,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                               is_best=is_best,
                               path=self.params.dict["root"],
                               filename = self.experimentName)

    def loadModel(self, model, optimizer, is_best):
        # Save weights
        return utils.load_checkpoint(path     = self.params.dict["root"],
                        filename = self.experimentName,
                        model    = model,
                        optimizer = optimizer)


    def setSeeds(self, seed):

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)

    def greedy(self, current_state, model:net.DeepQLearner):
        return self.greedyScheme(current_state, model)



    def _greedy(self, current_state, model:net.DeepQLearner):
        model.eval()
        with torch.no_grad():
            Q_values = model(current_state)
        model.train()

        best_action = np.argmax(Q_values.cpu().data.numpy())
        return best_action


    def _greedy_softmax(self, current_state, model:net.DeepQLearner):
        model.eval()
        with torch.no_grad():
            Q_values = model(current_state)
        model.train()
        # Choose the best action
        soft_max = torch.nn.Softmax(dim=0)
        probs = soft_max(Q_values.cpu()).data.numpy()
        soft_max_action = np.random.choice(range(self.actions_dim), p=probs)
        return soft_max_action

    def performEpsilonGreedy(self, current_state, model:net.DeepQLearner):
        # choose an action based on epsilon-greedy algorithm
        if np.random.rand() < self.epsilon:
            action = np.random.choice(range(self.actions_dim))
        else:
            action = self.greedy(current_state, model)
        return action


    def runReplayEpisodes(self, model:net.DeepQLearner):
        episode_terminated = False
        # Set full exploration
        self.epsilon = 1.0
        for _ in range(self.params.dict["replay_size"]):
            current_state = self.env.reset()
            while not episode_terminated:
                current_state, reward, episode_terminated, add_info = self.runSingleEpisode(current_state, model)


    def runSingleEpisode(self, current_state, model:net.DeepQLearner):

        current_state_tensor = torch.from_numpy(current_state).float().to(self.device)
        action = self.performEpsilonGreedy(current_state_tensor, model)
        # try:
        #     action = self.performEpsilonGreedy(current_state_tensor, model)
        # except:
        #     print("Exception")
        #     pdb.set_trace()
        next_state, reward, episode_terminated, add_info = self.env.step(action)
        # Add this to the replay, S->A->R->S', bolean to see if the episode terminated, and discard the addtional info
        # A' will be given by the neural net in a subsequent step
        self.replay.insert((current_state, action, reward, next_state, episode_terminated))
        # Insert it into the epsilon scheduler, as some schemes require it
        self.epsilonScheduler.updateRewards(reward)
        return next_state, reward, episode_terminated, add_info

    def runSingleEpisodeTestSet(self, env_test, current_state, model:net.DeepQLearner):

        current_state_tensor = torch.from_numpy(current_state).float().to(self.device)
        action = self.greedy(current_state_tensor, model)
        next_state, reward, episode_terminated, add_info = env_test.step(action)
        return next_state, reward, episode_terminated, add_info

    def getOneHotVectorForActions(self, batchSize, current_actions, numActions):
        """
        Helper function that creates a mask to select the actions that were experienced from all the action values returned by the network
        """
        actions_onehot = torch.FloatTensor(batchSize, numActions).zero_()
        actions = torch.LongTensor(current_actions).view(-1, 1)
        actions_onehot.scatter_(1, actions, 1)
        return actions_onehot


    def fillUpReplayBuffer(self, model:net.DeepQLearner):
        """
        Fills up the replay buffer with samples from the neutal network model, using epsilon greedy
        """
        episode_terminated = False
        # Set full exploration
        self.epsilon = 1.0
        for _ in range(self.params.dict["replay_size"]):
            current_state = self.env.reset()
            while not episode_terminated:
                current_state, reward, episode_terminated, add_info = self.runSingleEpisode(current_state,
                                                                                            model = model) # For Double deep Q-learning, select the training model
    def getMaxQOverA(self, Q_Sprime_APrime):
        #pdb.set_trace()
        Q_S_Amax, _ =  torch.max(Q_Sprime_APrime, axis = 1)
        return Q_S_Amax

    def runCalibratedPolicy(self, neuralNet:net.DeepQLearner, saveFrames = True, closeEnvironment = False):

        frames_one_episode = []

        total_reward = 0

        current_state = self.env.reset()
        episode_terminated = False
        while not episode_terminated:
            if saveFrames:
                frames_one_episode.append(self.env.render(mode="rgb_array"))

            best_action = self.greedy(current_state, neuralNet)
            current_state, reward, episode_terminated, add_info = self.env.step(best_action)

            total_reward += reward
        if closeEnvironment:
            # Close the environment, otherwise, an animation will pop up and freeze Python
            self.env.close()
        return total_reward, frames_one_episode

    def runCalibratedPolicyAndSaveResults(self, running_training_rewards, root, neuralNet:net.DeepQLearner, saveFrames = True, closeEnvironment = False,
                                          filename_animation = None, filename_rewards_plot = None, title = None, rolling_window = None):

        #total_rewards, simulation_frames = self.runCalibratedPolicy(neuralNet, saveFrames, closeEnvironment)
        #utils.saveFramesIntoAnnimation(simulation_frames, root, self.params, filename_animation)

        utils.plotRewardsPerEpisode(rewards_per_episode = running_training_rewards,
                                    configParams = self.params,
                                    ax = None,
                                    title = title,
                                    rolling_window = rolling_window,
                                    filename =filename_rewards_plot
                                    )

    def runTestEpisodes(self, model):

        self.numTestRuns += 1

        # Create a fresh environment

        env_test = gym.make(self.envName)

        env_solved = False
        made_progress = True

        time_start = time.time()
        test_rewards = []

        moving_ave_rewards = deque(maxlen=5)
        envSolvedCriteria  = EnvironmentSolved(env_test)


        pbar = tqdm(range(self.params.dict['nEpisodesTestRun']))



        # with profiler.profile(profile_memory = True, record_shapes = True) as prof:
        for episode_idx in pbar:
            episode_terminated = False
            episode_rewards = 0
            step_count = 0
            current_state = env_test.reset()
            while not episode_terminated:
                # Run one episode, over-write the current state with the next state
                with torch.no_grad():
                    current_state, reward, episode_terminated, add_info = self.runSingleEpisodeTestSet(env_test, current_state,
                                                                                            model=model)
                episode_rewards += reward


            test_rewards.append(episode_rewards)
            envSolvedCriteria.insertReward(episode_rewards)
            moving_ave_rewards.append(episode_rewards)
            running_time = time.time()


            pbar.set_postfix({"TEST RUN - Total Episode Rewards ": np.mean(moving_ave_rewards), "Episode ": episode_idx,
                              "Num Params Training ": self.count_parameters(self.trainingModel),
                              "Num params target ": self.count_parameters(self.targetModel),
                              "Gradient descend calls": self.gradientCallCount,
                              "Runnign time": running_time - time_start
                              }
                             )


            if episode_idx > envSolvedCriteria.numEpisodesWindow:
                if envSolvedCriteria.episodeSolved():
                    print("* * * ENVIRONMENT SOLVED! * * *")
                    self.savePlots(episode_idx, self.experimentName + " TEST RUN", test_rewards)
                    env_solved = True
                    made_progress = True
                    break

        self.test_run_mean_rewards.append(np.mean(test_rewards))

        if self.numTestRuns >= self.params.dict["made_progress_after_runs"] and not env_solved:
            if len(self.test_run_mean_rewards)>1:
                if self.test_run_mean_rewards[-1]> self.test_run_mean_rewards[0]:
                    made_progress = True
                else:
                    made_progress = False
            self.savePlots(episode_idx, self.experimentName + " TEST RUN - Not Solved", test_rewards)
            return made_progress, env_solved, test_rewards


        return made_progress, env_solved, test_rewards





    @abstractmethod
    def learningUpdate(self):
        pass

    @abstractmethod
    def runEpisodes(self):
        pass



class DeepQLearning(DeepQ):

    def __init__(self, params:ConfigParams, env:gym.Env, neuralNetModel: net.DeepQLearner, device = torch.device("cpu"), experimentName = None):
        super(__class__, self).__init__(env, params, device, experimentName)



        self.gamma = params.dict["gamma"]

        self.obs_dim                              = env.observation_space.shape[0]
        self.actions_dim                          = env.action_space.n


        self.replay = Replay(replay_size = params.dict["replay_size"], samplingMethod=params.dict["samplingMethod"], device = device)

        self.epsilon = None # This will be set by the function updateEpsilon

        self.neuralNetModel = neuralNetModel

        self.numTestRuns = 0




    def learningUpdate(self, batchSize, step_count):
        # Sample from the replay
        replay_data = self.replay.sample(batchSize)
        current_states, current_actions, rewards, next_sates, episode_terminateds = replay_data

        ####################
        # Q-learning target:
        ####################
        # Get Q_Sprime, taking the max over all actions:
        with torch.no_grad():
            #Q_Sprime_APrime = self.neuralNetModel(next_sates).detach()
            Q_Sprime_APrime = self.neuralNetModel(next_sates)
            Q_Sprime_Max_A = self.getMaxQOverA(Q_Sprime_APrime)

            # Only add gamma * Q_Sprime_Max_A for those episodes that did not terminate, so that episode_terminateds = 0
            # Distringuish between episodes that terminated because it reached the maximum number of steps, or because of a crash
            if self.env._max_episode_steps==step_count:
                # Do not punish this Q-value, since the episode finished becasue it reached the maximum number of steps
                Q_Learning_target = rewards + self.gamma * Q_Sprime_Max_A
            else:
                # Terminal state is a crash, zero-out this Q-value
                Q_Learning_target = rewards + self.gamma * Q_Sprime_Max_A * (
                            1 - episode_terminateds)

        # Q-learning update from the training model:
        # Current Q-values
        Q_S_all_actions = self.neuralNetModel(current_states)
        # Only keep the vallues from Q that correspond to the actions that occurred with the current_states
        actions_onehot = self.getOneHotVectorForActions(batchSize, current_actions, self.actions_dim)
        Q_S_A = torch.sum(actions_onehot * Q_S_all_actions, dim=1, keepdim=False)

        # Perform one step of the optimization
        self.neuralNetModel.gradientDescend(Q_S_A, Q_Learning_target)

    def runEpisodes(self, experimentName = None):

        episode_intervals_to_save_results = [int(interval * self.params.dict["nEpisodes"]) for interval in self.params.dict["saveResultIntervals"]]

        training_rewards = []

        env_solved = False
        made_progress = True

        # set model to training mode
        self.neuralNetModel.train()

        if self.params.dict["fillUpReplayBufferAtStart"]:
            self.runReplayEpisodes(model = self.neuralNetModel) # Fill up the replay buffer with the training model samples at the start

        #with tqdm(total=params.dict["nEpisodes"], mininterval=60.0) as pbar:
        for episode_idx in tqdm(range(self.params.dict["nEpisodes"])):
            episode_terminated = False
            episode_rewards = 0
            step_count      = 1
            current_state = self.env.reset()
            while not episode_terminated:
                self.epsilon = self.epsilonScheduler.getEpsilon(episode_idx = episode_idx)
                #Run one episode, over-write the current state with the next state
                current_state, reward, episode_terminated, add_info = self.runSingleEpisode(current_state, self.neuralNetModel)
                episode_rewards += reward

                step_count += 1

                if episode_idx >= self.params.dict["minEpisodesToTrainNet"]:
                    if step_count % self.params.dict["trainEveryStep"] == 0:
                        self.learningUpdate(self.params.dict["batchSize"], step_count)





            training_rewards.append(episode_rewards)

            # Check to see if you need to save intermediate results
            if (episode_idx+1) in episode_intervals_to_save_results:
                filename_rewards_plot, extension_plot = self.params.dict["fig_name"].split(".")
                filename_rewards_plot = filename_rewards_plot + " - Experiment {} - at Episode {:.0f}.{}".format(
                    experimentName, episode_idx + 1, extension_plot)

                filename_animation, _ = self.params.dict["fig_name"].split(".")
                filename_animation = filename_animation + " Experiment {} at Episode {:.0f} - .gif".format(
                    experimentName,
                    episode_idx + 1)

                fig_title             =   self.params.dict["fig_title"] + " - Experiment {}".format(experimentName)
                self.runCalibratedPolicyAndSaveResults(training_rewards, self.params.dict["root"],
                                              self.neuralNetModel, saveFrames = True, closeEnvironment = True,
                                        filename_animation = filename_animation, filename_rewards_plot = filename_rewards_plot,
                                                       title = fig_title, rolling_window = self.params.dict["rolling_window"])

            # Perform a test run and see if the environment was solved
            if (episode_idx+1)%self.params.dict["runTestEveryEpisode"]==0:
                made_progress, env_solved, test_run_rewards = self.runTestEpisodes(self.neuralNetModel)
                if env_solved or not made_progress:
                    if not made_progress:
                        print("Agent did not made progress after {} test runs... quitting".format(self.params.dict["made_progress_after_runs"]))
                    break
        return env_solved, made_progress, training_rewards, test_run_rewards








class DoubleDeepQLearning(DeepQ):

    def __init__(self, params: ConfigParams, env: gym.Env, neuralNetModel: net.DeepQLearner, experimentName = None, device = torch.device("cpu")):
        super(__class__, self).__init__(env, params, device, experimentName)

        self.gamma = params.dict["gamma"]

        self.obs_dim = env.observation_space.shape[0]
        self.actions_dim = env.action_space.n
        self.minEpisodesFullExploration = params.dict[
            "minEpisodesFullExploration"]  # How many episodes will it run with 100% exploration

        self.replay = Replay(replay_size=params.dict["replay_size"], samplingMethod=params.dict["samplingMethod"], device = device)

        self.epsilon = None  # This will be set by the function updateEpsilon

        self.neural_net_update_weight = self.params.dict["neural_net_update_weight"]


        self.trainingModel = deepcopy(neuralNetModel).to(self.device)
        self.targetModel = deepcopy(neuralNetModel).to(self.device)

        #self.trainingModel = net.DuelingDeepQLearnerCustomizable(obs_dim=self.obs_dim , action_dim=self.actions_dim, params=params)
        #self.targetModel   = net.DuelingDeepQLearnerCustomizable(obs_dim=self.obs_dim , action_dim=self.actions_dim, params=params)

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(params=self.trainingModel.parameters(),
                                          lr=params.dict['pytorch_optim_learning_rate'])

        # Initialize the Loss function

        loss_function_name = params.dict["neural_network_loss_func"]
        if loss_function_name == "MSE":
            self.criteria = torch.nn.MSELoss()
        elif loss_function_name == "RMSE":
            self.criteria = torch.nn.RMSELoss()
        elif loss_function_name == "L1Loss":
            self.criteria = torch.nn.L1Loss()
        else:
            raise ValueError("Loss {} not recognized!".format(loss_function_name))

        self.gradientClipValue = params.dict["gradientClipValue"]


        self.updateTargetModelFunction = self.weightedUpdateTargetModel if self.params.dict["updateTargetModel"]=="soft" else self.updateTargetModel

        self.gradientCallCount = 0

        self.expName = experimentName

    def learningUpdate(self, batchSize, step_count):
        # Sample from the replay
        replay_data = self.replay.sample(batchSize)
        current_states, current_actions, rewards, next_sates, episode_terminateds = replay_data

        with torch.no_grad():
            # Q-learning target
            # Obtain the values of Q across all actions
            #Q_S_next_A_all = self.trainingModel(next_sates).detach()
            Q_S_next_A_all = self.trainingModel(next_sates)
            #_, best_action = torch.max(Q_S_next_A_all, axis=1)
            _, best_action = Q_S_next_A_all.max(1)

            #best_action_mask = self.getOneHotVectorForActions(batchSize, best_action, self.actions_dim)
            #Q_SPrime_Max_A = torch.sum(best_action_mask * self.targetModel(next_sates).detach(), dim=1, keepdim=False)

            #Q_SPrime_Max_A = self.targetModel(next_sates).detach().gather(1, best_action.unsqueeze(1))
            Q_SPrime_Max_A = self.targetModel(next_sates).gather(1, best_action.unsqueeze(1))
            # Distringuish between episodes that terminated because it reached the maximum number of steps, or because of a crash
            if self.env._max_episode_steps == step_count:
                # Do not punish this Q-value, since the episode finished becasue it reached the maximum number of steps
                Q_Learning_target = rewards + self.gamma * Q_SPrime_Max_A
            else:
                Q_Learning_target = rewards + self.gamma * Q_SPrime_Max_A * (
                    1 - episode_terminateds)

        # Current Q-values
        Q_S_all_actions = self.trainingModel(current_states)
        # Only keep the vallues from Q that correspond to the actions that occurred with the current_states
        #actions_onehot = self.getOneHotVectorForActions(batchSize, current_actions, self.actions_dim)
        #Q_S_A = torch.sum(actions_onehot * Q_S_all_actions, dim=1, keepdim=False)
        Q_S_A = Q_S_all_actions.gather(1, current_actions.unsqueeze(1))


        # Perform one step of the optimization
        self.gradientDescend(Q_S_A, Q_Learning_target)

        # self.gradientCallCount += 1
        # # loss = self.criteria(Q_S_A, Q_Learning_target)
        # loss = F.mse_loss(Q_S_A, Q_Learning_target)
        # # Train the net
        # # self.optimizer.zero_grad()
        # # loss.backward()
        # # if self.gradientClipValue is not None:
        # #     for param in self.trainingModel.parameters():
        # #         param.grad.data.clamp_(-self.gradientClipValue, self.gradientClipValue)
        # # for _ in range(self.params.dict["additionalTrainingLoop"]):
        # #     self.optimizer.step()
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

    def updateTargetModel(self):
        # Update the target model with the parameters from the training model
        self.targetModel.load_state_dict(self.trainingModel.state_dict())

    def weightedUpdateTargetModel(self):
        # Do a weighted average of the target model with the parameters from the training model
        for target_param, training_param in zip(self.targetModel.parameters(), self.trainingModel.parameters()):
            target_param.data.copy_(self.neural_net_update_weight*training_param.data +
                                        (1.0-self.neural_net_update_weight)*target_param.data)


    def gradientDescend(self, Q_S_A, Q_Learning_target):
        self.gradientCallCount += 1
        loss = self.criteria(Q_S_A, Q_Learning_target)
        #loss = F.mse_loss(Q_S_A, Q_Learning_target)
        # Train the net
        # self.optimizer.zero_grad()
        # loss.backward()
        # if self.gradientClipValue is not None:
        #     for param in self.trainingModel.parameters():
        #         param.grad.data.clamp_(-self.gradientClipValue, self.gradientClipValue)
        # for _ in range(self.params.dict["additionalTrainingLoop"]):
        #     self.optimizer.step()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def savePlots(self, episode_idx, experimentName, total_rewards):

        filename_rewards_plot, extension_plot = self.params.dict["fig_name"].split(".")
        filename_rewards_plot = filename_rewards_plot + " - Experiment {} - at Episode {:.0f}.{}".format(
            experimentName, episode_idx + 1, extension_plot)

        filename_animation, _ = self.params.dict["fig_name"].split(".")
        filename_animation = filename_animation + " Experiment {} at Episode {:.0f} - .gif".format(
            experimentName,
            episode_idx + 1)

        fig_title = self.params.dict["fig_title"] + " - Experiment {}".format(experimentName)
        self.runCalibratedPolicyAndSaveResults(total_rewards,
                                               self.params.dict["root"],
                                               self.trainingModel, saveFrames=True, closeEnvironment=True,
                                               filename_animation=filename_animation,
                                               filename_rewards_plot=filename_rewards_plot,
                                               title=fig_title,
                                               rolling_window=self.params.dict["rolling_window"])

    def runEpisodes(self, experimentName = None):

        time_start = time.time()

        episode_intervals_to_save_results = [int(interval * self.params.dict["nEpisodes"]) for interval in
                                             self.params.dict["saveResultIntervals"]]


        training_rewards = []

        if self.params.dict["fillUpReplayBufferAtStart"]:
            self.runReplayEpisodes(model = self.trainingModel) # Fill up the replay buffer with the training model samples at the start

        # set model to training mode
        #self.trainingModel.train()

        moving_ave_rewards = deque(maxlen = 100)
        best_average_reward        = -1e10

        env_solved = False
        made_progress = True

        # with tqdm(total=params.dict["nEpisodes"], mininterval=60.0) as pbar:
        pbar = tqdm(range(self.params.dict["nEpisodes"]))

        #with profiler.profile(profile_memory = True, record_shapes = True) as prof:
        for episode_idx in pbar:
            episode_terminated = False
            episode_rewards = 0
            step_count = 0
            current_state = self.env.reset()
            while not episode_terminated:
            #for t in range(1000):
                self.epsilon = self.epsilonScheduler.getEpsilon(episode_idx=episode_idx)
                # Run one episode, over-write the current state with the next state
                with torch.no_grad():
                    current_state, reward, episode_terminated, add_info = self.runSingleEpisode(current_state, model=self.trainingModel)
                episode_rewards += reward

                step_count += 1

                if episode_idx >= self.params.dict["minEpisodesToTrainNet"]:
                    if step_count%self.params.dict["trainEveryStep"]==0:
                        self.learningUpdate(self.params.dict["batchSize"], step_count)

                if episode_idx % self.params.dict["numEpisodesToUpdateTargetModel"] == 0:
                    self.updateTargetModelFunction()


            training_rewards.append(episode_rewards)

            # Check to see if you need to save intermediate results
            if (episode_idx+1) in episode_intervals_to_save_results:
                self.savePlots(episode_idx, experimentName, training_rewards)

            moving_ave_rewards.append(episode_rewards)
            running_time = time.time()
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_idx, np.mean(moving_ave_rewards)), end="")
            if (episode_idx) % 20 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}\tRunning time: {:.2f}'.format(episode_idx, np.mean(moving_ave_rewards), running_time - time_start))

            self.count_parameters(self.trainingModel)



            pbar.set_postfix({"Total Episode Rewards ": np.mean(moving_ave_rewards), "Episode ": episode_idx,
                              "Num Params Training ": self.count_parameters(self.trainingModel),
                                "Num params target ": self.count_parameters(self.targetModel),
                             "Gradient descend calls": self.gradientCallCount,
                              "Runnign time": running_time - time_start
                              }
                             )

            if episode_idx%20==0:
                torch.cuda.empty_cache()

            mean_rewards = np.mean(moving_ave_rewards)
            # Save the model and optimizer
            self.saveModel(model = self.trainingModel, episode_idx = episode_idx, optimizer=self.optimizer,
                           is_best= best_average_reward < mean_rewards)

            if best_average_reward < mean_rewards:
                best_average_reward = mean_rewards

            # Perform a test run and see if the environment was solved
            if (episode_idx + 1) % self.params.dict["runTestEveryEpisode"] == 0:
                made_progress, env_solved, test_run_rewards = self.runTestEpisodes(self.trainingModel)
                if env_solved or not made_progress:
                    if not made_progress:
                        print("Agent did not made progress after {} test runs... quitting".format(
                            self.params.dict["made_progress_after_runs"]))
                    break

        return env_solved, made_progress, training_rewards, test_run_rewards

    #def runCalibratedPolicy(self, saveFrames = True):
    #    return DeepQ.runCalibratedPolicy(self, neuralNet = self.trainingModel, saveFrames = saveFrames)

    def count_parameters(self, model):
        #return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())

class SarsaQLearning(DeepQ):

    def __init__(self, params: ConfigParams, env: gym.Env, neuralNetModel: net.DeepQLearnerCustomizable, device = torch.device("cpu"), experimentName = None):
        super(__class__, self).__init__(env, params, device, experimentName)

        self.gamma = params.dict["gamma"]

        self.obs_dim = env.observation_space.shape[0]
        self.actions_dim = env.action_space.n
        self.minEpisodesFullExploration = params.dict[
            "minEpisodesFullExploration"]  # How many episodes will it run with 100% exploration

        self.epsilon = None  # This will be set by the function updateEpsilon


        self.trainingModel = deepcopy(neuralNetModel)

        self.greedySchemeSarsa = self._greedy_softmax_sarsa if params.dict["greedy_scheme"] == "softmax" else self._greedy_sarsa



    def performEpsilonGreedy_Sarsa(self, next_state, model:net.DeepQLearner):
        # choose an action based on epsilon-greedy algorithm
        if np.random.rand() < self.epsilon:
            action = np.random.choice(range(self.actions_dim))
        else:
            action = self.greedySarsa(next_state, model)
        return action

    def greedySarsa(self, next_states, model: net.DeepQLearner):
        return self.greedySchemeSarsa(next_states, model)

    def _greedy_sarsa(self, next_states, model: net.DeepQLearner):
        Q_values = model(next_states)
        with torch.no_grad():
        #best_action = torch.argmax(Q_values.detach())
            best_action = torch.argmax(Q_values)
        return int(best_action.numpy())



    def _greedy_softmax(self, next_states, model: net.DeepQLearner):
        with torch.no_grad():
            # Choose the best action
            soft_max = torch.nn.Softmax(dim=0)
            #probs = soft_max(self.trainingModel(next_states)).detach().numpy()
            probs = soft_max(self.trainingModel(next_states)).numpy()
            soft_max_actions = [np.random.choice(range(self.actions_dim), p=p) for p in probs]
        return soft_max_actions


    def learningUpdate(self, batchSize, current_state, current_action, reward, next_state, next_action, episode_terminated,
                       step_count):


        # Sarsa learning target

        # Computation of Q(S,A)
        #######################

        Q_S_all_actions = self.trainingModel(current_state)

        Q_S_A = Q_S_all_actions[current_action]


        with torch.no_grad():
            # Computation of Q(S',A')
            #######################
            # Do not perform gradient descent for this one... detach it.
            #Q_S_Prime_A_Prime_actions = self.trainingModel(next_state).detach()
            Q_S_Prime_A_Prime_actions = self.trainingModel(next_state)
            Q_S_Prime_A_prime = Q_S_Prime_A_Prime_actions[next_action]

            # Distringuish between episodes that terminated because it reached the maximum number of steps, or because of a crash
            if self.env._max_episode_steps == step_count:
                # Do not punish this Q-value, since the episode finished becasue it reached the maximum number of steps
                Q_Learning_target = torch.as_tensor(reward) + self.gamma * Q_S_Prime_A_prime

            else:
                # Terminal state is a crash, zero-out this Q-value
                Q_Learning_target = torch.as_tensor(reward) + self.gamma * Q_S_Prime_A_prime * (
                    1 - torch.as_tensor(episode_terminated).float())

        # Perform one step of the optimization
        self.trainingModel.gradientDescend(Q_S_A.double(), Q_Learning_target.double())

    def runEpisodes(self, experimentName=None):

        episode_intervals_to_save_results = [int(interval * self.params.dict["nEpisodes"]) for interval in
                                             self.params.dict["saveResultIntervals"]]

        env_solved = False
        made_progress = True


        training_rewards = []

        # set model to training mode
        self.trainingModel.train()

        # with tqdm(total=params.dict["nEpisodes"], mininterval=60.0) as pbar:
        for episode_idx in tqdm(range(self.params.dict["nEpisodes"])):
            episode_terminated = False
            episode_rewards = 0
            step_count      = 1
            current_state = self.env.reset()
            while not episode_terminated:
                self.epsilon = self.epsilonScheduler.getEpsilon(episode_idx=episode_idx)
                current_action = self.performEpsilonGreedy_Sarsa(current_state, self.trainingModel)
                next_state, reward, episode_terminated, add_info = self.env.step(current_action)
                # Insert it into the epsilon scheduler, as some schemes require it
                self.epsilonScheduler.updateRewards(reward)
                # Take an epsilon greedy action
                with torch.no_grad():
                    next_action = self.performEpsilonGreedy(next_state, self.trainingModel)



                episode_rewards += reward

                self.learningUpdate(self.params.dict["batchSize"],
                                    current_state, current_action, reward, next_state, next_action, episode_terminated, step_count)
                step_count += 1

                # Check to see if you need to save intermediate results
                if (episode_idx+1) in episode_intervals_to_save_results:
                    filename_rewards_plot, extension_plot = self.params.dict["fig_name"].split(".")
                    filename_rewards_plot = filename_rewards_plot + " - Experiment {} - at Episode {:.0f}.{}".format(
                        experimentName, episode_idx + 1, extension_plot)

                    filename_animation, _ = self.params.dict["fig_name"].split(".")
                    filename_animation = filename_animation + " Experiment {} at Episode {:.0f} - .gif".format(
                        experimentName,
                        episode_idx + 1)

                    fig_title = self.params.dict["fig_title"] + " - Experiment {}".format(experimentName)
                    self.runCalibratedPolicyAndSaveResults(training_rewards,
                                                           self.params.dict["root"],
                                                           self.trainingModel, saveFrames=True, closeEnvironment=True,
                                                           filename_animation=filename_animation,
                                                           filename_rewards_plot=filename_rewards_plot,
                                                           title=fig_title,
                                                           rolling_window=self.params.dict["rolling_window"])



            training_rewards.append(episode_rewards)

            # Perform a test run and see if the environment was solved
            if (episode_idx + 1) % self.params.dict["runTestEveryEpisode"] == 0:
                made_progress, env_solved, test_run_rewards = self.runTestEpisodes(self.neuralNetModel)
                if env_solved or not made_progress:
                    if not made_progress:
                        print("Agent did not made progress after {} test runs... quitting".format(
                            self.params.dict["made_progress_after_runs"]))
                    break

        return env_solved, made_progress,training_rewards, test_run_rewards

    #def runCalibratedPolicy(self, saveFrames = True):
    #    return DeepQ.runCalibratedPolicy(self, neuralNet = self.trainingModel, saveFrames = saveFrames)

class EnvironmentSolved():

    def __init__(self, env, numEpisodesWindow = 100):
        self.rewards = []
        self.env = env
        self.numEpisodesWindow = numEpisodesWindow

    def insertReward(self, rewards:list):
        self.rewards.append(rewards)

    def reset(self):
        self.rewards = []

    def episodeSolved(self, threshold = 200):
        windowed_rewards = pd.Series(self.rewards).rolling(window=self.numEpisodesWindow).mean().dropna().iloc[-1]
        return windowed_rewards >= threshold

    def testRunSolved(self):
        # Reset environment
        self.env.reset()






class PerformanceStatistics():

    def __init__(self):

        self.rewards = []

    def updateRewards(self, rewards:list):
        self.rewards = deepcopy(rewards)

    def insertReward(self, reward):
        self.rewards.append(reward)

    def statistics(self, window):
        rolling = pd.Series(self.rewards).rolling(window=window)
        return {"mean": np.mean(self.rewards), "std": np.std(self.rewards), "rolling mean": rolling.mean(), "rolling std": rolling.std()}

class GridSearch():

    def __init__(self, baseParams: ConfigParams, envName, searchParamsDict:dict, device = torch.device("cpu")):

        self.baseParams           = deepcopy(baseParams)
        self.envName              = envName
        self.searchParamsDict     = searchParamsDict

        self.searchSpace = {}
        self.setSearchSpace = set()
        self.counter     = 1
        self.params = baseParams
        self.device = device

    def searchEpisodeName(self):
        name = "Exp {}".format(self.counter)
        self.counter += 1
        return name

    def reset(self, newParams: ConfigParams):
        # Reset the environment
        self.env = gym.make(self.envName)
        self.env.seed(self.baseParams.dict["RANDOM_SEED"])
        self.env.action_space.seed(self.baseParams.dict["RANDOM_SEED"])

        # Initialize the neural network
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        # Initialize the neural network
        if newParams.dict["NeuralNet"] == "DeepQLearnerCustomizable":
            neuralnet = net.DeepQLearnerCustomizable(obs_dim=obs_dim, action_dim=action_dim, params=newParams, device = self.device)
        elif newParams.dict["NeuralNet"] == "DuelingDeepQLearnerCustomizable":
            neuralnet = net.DuelingDeepQLearnerCustomizable(obs_dim=obs_dim, action_dim=action_dim, params=newParams, device = self.device)

        # Initialize the RL agent
        if newParams.dict["RLAgent"] == "DeepQLearner":
            self.agent = DeepQLearning(params=newParams, env=self.env, neuralNetModel = neuralnet, device = self.device)
        elif newParams.dict["RLAgent"] == "DoubleDeepQLearner":
            self.agent = DoubleDeepQLearning(params=newParams, env=self.env, neuralNetModel = neuralnet, device = self.device)
        elif newParams.dict["RLAgent"] == "SarsaQLearning":
            self.agent = SarsaQLearning(params=newParams, env=self.env, neuralNetModel = neuralnet, device = self.device)
        else:
            raise ValueError("Agent {} not recognized!".format(newParams.dict["RLAgent"]))



    def getHyperParameterSearchSpace(self):
        #Make a copy, using the first element of setOfRuns
        runningConfig = deepcopy(self.baseParams)
        for key, values in self.searchParamsDict.items():
            runningConfig.dict[key] = values[0]
        # Set a container for the list of all the runs that we will do
        allRuns = [runningConfig]
        # Now iterate
        paramsToSearch = list(self.searchParamsDict.keys())
        return self.gridLoop(allRuns, self.searchParamsDict, paramsToSearch)


    def gridLoop(self, allRuns, setOfRuns, paramsToSearch):
        if len(paramsToSearch)>0:
            param_to_loop = paramsToSearch[-1]
            newRuns = []
            #pdb.set_trace()
            for runconfig in allRuns:
                for val in setOfRuns[param_to_loop]:
                    runconfig.dict[param_to_loop] = val
                    newRuns.append(deepcopy(runconfig))
            # Add it to allRuns

            if len(allRuns)==1:

                allRuns = newRuns
            else:

                #pdb.set_trace()
                allRuns.extend(newRuns)

            # Call it again
            return self.gridLoop(allRuns, setOfRuns, paramsToSearch[:-1])
        else:
            return allRuns


    def cleanupList(self, listOfRuns):
        # Gets rid of duplicates
        uniqueRuns = list(set(json.dumps(r.dict) for r in listOfRuns))
        jobs = {}
        self.counter=1
        for job in uniqueRuns:
            runName = self.searchEpisodeName()
            jobs[runName] = ConfigParams(json_path=None, jsondata = json.loads(job))
        return jobs

    def runOneJob(self, job, exp_name = None):
        if isinstance(job, dict):
            newjob = ConfigParams(json_path=None, jsondata=json(job))
        else:
            newjob = job
        # Reset
        self.reset(newParams=newjob)
        # Reset seeds
        self.agent.setSeeds(self.params.dict["RANDOM_SEED"])
        # Set the name of the experiment in the agent so it can save the network params
        self.agent.experimentName = exp_name
        # Run the agent
        agent_data = self.agent.runEpisodes(experimentName=exp_name)
        return agent_data

    def saveJobsDictToFile(self, jobsDict):
        if "jobs_filename" in self.params.dict:
            pp({exp_name: j.dict for exp_name, j in jobsDict.items()}, open(os.path.join(self.params.dict["root"], self.params.dict["jobs_filename"]), "w"))

    def loadJobsCompleted(self):
        filename = os.path.join(self.params.dict["root"], self.params.dict["GridJobsCompletedFileName"])
        #return json.load(filename)
        jobs_done = pickle.load(open(filename, "rb"))
        print("Loaded {} jobs done...".format(len(jobs_done.keys())))
        return jobs_done

    def anyJobsCompleted(self):
        filename = os.path.join(self.params.dict["root"], self.params.dict["GridJobsCompletedFileName"])
        return os.path.isfile(filename)

    def saveJobsCompleted(self, jobsCompleted: dict):
        filename = os.path.join(self.params.dict["root"], self.params.dict["GridJobsCompletedFileName"])
        #json.dump(json.dumps(jobsCompleted), filename)
        pickle.dump(jobsCompleted, open(filename, "wb"))
        print("Saving {} jobs done...".format(len(jobsCompleted.keys())))
        return

    def loadJobsToRun(self):

        jobs = self.getHyperParameterSearchSpace()
        jobs = self.cleanupList(jobs)

        jobs_completed = {}
        if self.anyJobsCompleted():
            jobs_completed = self.loadJobsCompleted()

        jobs_to_run = {}
        for experiment in jobs.keys():
            if experiment not in jobs_completed:
                jobs_to_run[experiment] = jobs[experiment]
        return jobs_to_run

    def runGrid(self):

        # Produce the runs
        jobs = self.loadJobsToRun()
        # save all the rewards
        completed_jobs = {}
        numJobs = len(jobs.keys())
        self.saveJobsDictToFile(jobs)
        # Start the loop
        runningtimes = []

        for idx, (exp_name, job) in enumerate(jobs.items()):
            tstart = time.time()
            print("Running . . . ", exp_name, " - job {}/{}".format(idx+1, numJobs))
            env_solved, made_progress, training_rewards, test_rewards = self.runOneJob(job, exp_name)
            #print("rewards_per_episode")
            #print(rewards_per_episode)
            completed_jobs[exp_name] = {"config": deepcopy(job), "training rewards": training_rewards, "test rewards": test_rewards,
                                        "env_solved": env_solved, "made_progress": made_progress}
            tend = time.time()
            runningtimes.append((tend - tstart)/(60.0))
            total_remain_time = np.mean(runningtimes) * (numJobs - idx - 1)
            print("Remaining time for completion. . . {:2.2f} min".format(total_remain_time))

            # Save the jobs completed so far
            self.saveJobsCompleted(completed_jobs)

        return completed_jobs

    def summarizeGridResults(self, gridResults, window = 10):
        stats = PerformanceStatistics()
        for expName, gridDict in gridResults.items():
            rewards = gridResults[expName]['rewards']
            stats.updateRewards(rewards)
            stats_data = stats.statistics(window)
            gridResults[expName].update(stats_data)
        # Sort gridResults
        return {exp: data for exp, data in sorted(gridResults.items(), key=lambda item: item[1]["rolling mean"].iloc[-1], reverse=True)}





