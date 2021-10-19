import torch.nn as nn
import torch
from utils.mathpy import *


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), activation='tanh', log_std=0):
        super(Policy, self).__init__()
        self.is_disc_action = False

        """ layers for inputs of depth_images """
        self.conv1 = nn.Conv2d(4, 32, (10, 14), (8, 8), padding=(1, 4))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), 2, padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1))
        self.fc_img = nn.Linear(8 * 10 * 64, 512)

        """ layers for inputs of goals """
        self.fc_goal = nn.Linear(4 * 2, 64)

        """ layers for inputs concatenated information """
        self.img_goal1 = nn.Linear(576, 512)
        self.img_goal2 = nn.Linear(512, action_dim) # two dimensions of actions: upward and downward; turning

        self.relu = nn.ReLU()
        self.img_goal2.weight.data.mul_(0.1)
        self.img_goal2.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, depth_img, goal): # remember to initialize
        # batch_size = x.size(0)
        depth_img = self.relu(self.conv1(depth_img))
        depth_img = self.relu(self.conv2(depth_img))
        depth_img = self.relu(self.conv3(depth_img))
        depth_img = depth_img.view(depth_img.size(0), -1)
        depth_img = self.relu(self.fc_img(depth_img))

        goal = goal.view(goal.size(0), -1)
        goal = self.relu(self.fc_goal(goal))

        img_goal = torch.cat((depth_img, goal), 1)
        img_goal = self.relu(self.img_goal1(img_goal))
        action_mean = self.relu(self.img_goal2(img_goal))

        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, depth_img, goal):
        action_mean, _, action_std = self.forward(depth_img, goal)
        action = torch.normal(action_mean, action_std)
        return action

    def get_log_prob(self, depth_img, goal, actions):
        action_mean, action_log_std, action_std = self.forward(depth_img, goal)
        return normal_log_density(actions, action_mean, action_log_std, action_std)