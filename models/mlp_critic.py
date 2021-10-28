import torch.nn as nn
import torch


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='tanh'):
        super(Value, self).__init__()
        """ layers for inputs of depth_images """
        self.conv1 = nn.Conv2d(4, 32, (10, 14), (8, 8), padding=(1, 4))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), 2, padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1))
        self.fc_img = nn.Linear(8 * 10 * 64, 512)

        """ layers for inputs of goals """
        self.fc_goal = nn.Linear(4 * 2, 64)

        """ layers for inputs concatenated information """
        self.img_goal1 = nn.Linear(576, 512)
        self.img_goal2 = nn.Linear(512, 1)  # two dimensions of actions: upward and downward; turning

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.img_goal2.weight.data.mul_(1)
        self.img_goal2.bias.data.mul_(0.0)

    def forward(self, depth_img, goal):
        depth_img = self.relu(self.conv1(depth_img))
        depth_img = self.relu(self.conv2(depth_img))
        depth_img = self.relu(self.conv3(depth_img))
        depth_img = depth_img.view(depth_img.size(0), -1)
        depth_img = self.relu(self.fc_img(depth_img))

        goal = goal.view(goal.size(0), -1)
        goal = self.relu(self.fc_goal(goal))

        img_goal = torch.cat((depth_img, goal), 1)
        img_goal = self.relu(self.img_goal1(img_goal))
        value = self.tanh(self.img_goal2(img_goal))

        return value
