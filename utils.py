import PIL
import os
import json
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
import shutil
import torch

class ConfigParams():
    """Class that loads hyperparameters from a json file.

    """

    def __init__(self, json_path, jsondata = None):
        self.paramdict = {}
        if jsondata is not None:
            params = jsondata
            self.paramdict.update(params)
        else:
            with open(json_path) as f:
                params = json.load(f)
                self.paramdict.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.paramdict , f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.paramdict.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.paramdict


def plotRewardsPerEpisode(rewards_per_episode, configParams:ConfigParams, ax = None, title = None, title_size = 16, legend_size = 16, axis_size = 16,
                          rolling_window = None, filename = None):
    """
    Perform plots of rewards
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    mean_reward = np.mean(rewards_per_episode)
    std_reward = np.std(rewards_per_episode)
    ax.plot(rewards_per_episode, alpha = 0.7)
    # Add rolling window average and stdev
    if rolling_window is not None:
        if rolling_window <= 1.0:
            rolling_window = int(configParams.dict["rolling_window"] * configParams.dict["nEpisodes"])
    else:
        if configParams.dict["rolling_window"] <= 1.0:
            rolling_window = int(configParams.dict["rolling_window"] * configParams.dict["nEpisodes"])
        else:
            rolling_window = configParams.dict["rolling_window"]


    rolling_average = pd.Series(rewards_per_episode).rolling(window = rolling_window).mean()
    rolling_std = pd.Series(rewards_per_episode).rolling(window=rolling_window).std()

    ax.set_xlabel("Episode", size=axis_size)
    ax.set_ylabel("Total reward per episode", size=axis_size)
    #ax.axhline(mean_reward, linestyle='--', color="black", label="Mean reward per episode {:2.1f}, ".format(mean_reward) + r"$\sigma = $" + "{:2.1f}".format(std_reward))
    #ax.plot(rolling_average, color = 'red', alpha = 0.7, label = "Rolling ave={:2.1f}".format(rolling_average.iloc[-1]) + r", $\sigma$ = " + "{:2.1f}".format(rolling_std.iloc[-1]))

    ax.axhline(mean_reward, linestyle='--', color="black", label=r"$R_{mean}$" + "=  {:2.1f}, ".format(mean_reward) + r"$\sigma = $" + "{:2.1f}".format(std_reward))
    ax.plot(rolling_average, color = 'red', alpha = 0.9, label = r"$R_{rolling}$"+"={:2.1f}".format(rolling_average.iloc[-1]) + r", $\sigma$ = " + "{:2.1f}".format(rolling_std.iloc[-1]))

    ax.legend(fontsize=legend_size)
    if title is None:
        ax.set_title("{} \n Reward across {:2.0f} episodes".format(configParams.dict["fig_title"],configParams.dict["nEpisodes"]),
                 size=title_size)
    else:
        ax.set_title(title, size=title_size)
    if ax is None or filename is not None:
        filename = filename if filename is not None else configParams.dict["fig_name"]
        fig.savefig(os.path.join(configParams.dict["root"], filename))
        plt.close()


def saveFramesIntoAnnimation(frames: list, root, configParams:ConfigParams, filename = None):
    """
    Saves a list of frames, given as np array of RGB format, into a gif annimation
    """
    images = []
    for i, frame in enumerate(frames):
        im = PIL.Image.fromarray(frame)
        images.append(im)
    filename = filename if filename is not None else configParams.dict["fig_name"].split(".")[0] + " - ANIMATION.gif"
    filename = os.path.join(root, filename)
    images[0].save(filename,
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)



def save_checkpoint(state, is_best, path, filename):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    """
    filepath = os.path.join(path, filename+'_last.pth.tar')
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(path, filename + '_best.pth.tar'))


def load_checkpoint(path, filename, model, optimizer):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    """
    filepath = os.path.join(path, filename + '_last.pth.tar')
    if not os.path.exists(filepath):
        raise Exception("File doesn't exist {}".format(filepath))
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def count_parameters(model):
    """
    Counts number of parameters in the neural network
    """
    return sum(p.numel() for p in model.parameters())