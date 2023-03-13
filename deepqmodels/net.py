import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb
from rlgtsrc.project2.utils import ConfigParams

torch.manual_seed(1234)
from copy import deepcopy

class DeepQLearner(torch.nn.Module):

    def __init__(self, obs_dim, action_dim, params:ConfigParams, device = torch.device("cpu")):
        super().__init__()

        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(obs_dim, params.dict["hidden_dim"]).to(device)
        self.fc2 = nn.Linear(params.dict["hidden_dim"], action_dim).to(device)

        self.gradientClipValue = params.dict["gradientClipValue"]

        self.tanh = torch.nn.Tanh()

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(params=self.parameters(),
                                     lr=params.dict['pytorch_optim_learning_rate'])

        loss_function_name = params.dict["neural_network_loss_func"]
        if loss_function_name == "RMSE":
            self.criteria = torch.nn.MSELoss()
        elif loss_function_name == "L1Loss":
            self.criteria = torch.nn.L1Loss()
        else:
            raise ValueError("Loss {} not recognized!".format(loss_function_name))



    def forward(self, obs):
        obs_tensor = torch.Tensor(obs)
        x = F.relu(self.fc1(obs_tensor))
        Q_values = self.fc2(x)
        return Q_values


    def gradientDescend(self, Q_S_A, Q_Learning_target):
        loss = self.criteria(Q_S_A, Q_Learning_target)
        # Train the net
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if self.gradientClipValue is not None:
            for param in self.parameters():
                param.grad.data.clamp_(-self.gradientClipValue, self.gradientClipValue)
        self.optimizer.step()



class DeepQLearnerCustomizable(torch.nn.Module):

    def __init__(self, obs_dim, action_dim, params:ConfigParams, device = torch.device("cpu")):
        super(DeepQLearnerCustomizable, self).__init__()

        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.layersdim  = params.dict["neuralNetDimensions"]
        if params.dict["activations"] == "relu":
            self.activation = torch.nn.ReLU()
        elif params.dict["activations"] == "tanh":
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError("Activation function {} not recognized!".format(params.dict["activations"]))
        blocks = [nn.Linear(obs_dim, self.layersdim[0]), self.activation]
        for ldim0, ldim1 in zip(self.layersdim, self.layersdim[1:]):
            blocks.extend([nn.Linear(ldim0, ldim1), self.activation])
        blocks.append(nn.Linear(self.layersdim[-1], action_dim))

        self.network = nn.Sequential(*blocks).to(device)

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(params=self.parameters(),
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

        self.params = deepcopy(params)

    def forward(self, obs):
        obs_tensor = torch.Tensor(obs)
        Q_values = self.network(obs_tensor)
        return Q_values


    def gradientDescend(self, Q_S_A, Q_Learning_target):
        loss = self.criteria(Q_S_A, Q_Learning_target)
        # Train the net
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if self.gradientClipValue is not None:
            for param in self.network.parameters():
                param.grad.data.clamp_(-self.gradientClipValue, self.gradientClipValue)
        for _ in range(self.params.dict["additionalTrainingLoop"]):
            self.optimizer.step()

class DuelingDeepQLearnerCustomizable(torch.nn.Module):

    def __init__(self, obs_dim, action_dim, params:ConfigParams, device = torch.device("cpu")):
        super(DuelingDeepQLearnerCustomizable, self).__init__()

        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.layersdim  = params.dict["neuralNetDimensions"]
        if params.dict["activations"] == "relu":
            self.activation = torch.nn.ReLU()
        elif params.dict["activations"] == "tanh":
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError("Activation function {} not recognized!".format(params.dict["activations"]))

        common_blocks = [nn.Linear(obs_dim, self.layersdim[0]), self.activation]

        for ldim0, ldim1 in zip(self.layersdim, self.layersdim[1:-1]):
            common_blocks.extend([nn.Linear(ldim0, ldim1), self.activation])


        value_block     = ([nn.Linear(self.layersdim[-2], self.layersdim[-1]), self.activation] +
                                                     [nn.Linear(self.layersdim[-1], 1), torch.nn.Identity()])

        advantage_block = ([nn.Linear(self.layersdim[-2], self.layersdim[-1]), self.activation] +
                                           [nn.Linear(self.layersdim[-1], action_dim), torch.nn.Identity()])


        self.common_network = nn.Sequential(*common_blocks).to(device)
        self.value_network = nn.Sequential(*value_block).to(device)
        self.advantage_network = nn.Sequential(*advantage_block).to(device)

   
    def forward(self, obs):

        x = self.common_network(obs)

        value_function = self.value_network(x)

        advantage_function = self.advantage_network(x)

        Q_value = value_function + advantage_function - advantage_function.mean().unsqueeze(0).expand(1, self.action_dim)[0]

        return Q_value


