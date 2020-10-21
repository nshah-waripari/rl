import torch.nn as nn
import torch.nn.functional as F



class ActionValueNetwork(nn.Module):

    def __init__(self, network_config):
        super(ActionValueNetwork, self).__init__()

        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")

        self.fc1 = nn.Linear(self.state_dim, self.num_hidden_units)
        nn.init.orthogonal_(self.fc1.weight)
        self.fc2 = nn.Linear(self.num_hidden_units, self.num_actions)
        nn.init.orthogonal_(self.fc2.weight)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



