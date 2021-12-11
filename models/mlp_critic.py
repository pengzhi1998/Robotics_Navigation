import torch.nn as nn
import torch


class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        """ layers for inputs of depth_images """
        self.conv1 = nn.Conv2d(4, 32, (10, 14), (8, 8), padding=(1, 4))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), 2, padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1))
        self.fc_img = nn.Linear(8 * 10 * 64, 512)

        """ layers for inputs of goals """
        self.fc_goal = nn.Linear(4 * 3, 96)
        self.fc_ray = nn.Linear(4 * 1, 32)

        """ layers for inputs concatenated information """
        self.img_goal_ray1 = nn.Linear(640, 512)
        self.img_goal_ray2 = nn.Linear(512, 1)  # two dimensions of actions: upward and downward; turning

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.img_goal_ray2.weight.data.mul_(1)
        self.img_goal_ray2.bias.data.mul_(0.0)

    def forward(self, depth_img, goal, ray):
        depth_img = self.relu(self.conv1(depth_img))
        depth_img = self.relu(self.conv2(depth_img))
        depth_img = self.relu(self.conv3(depth_img))
        depth_img = depth_img.view(depth_img.size(0), -1)
        depth_img = self.relu(self.fc_img(depth_img))

        goal = goal.view(goal.size(0), -1)
        goal = self.relu(self.fc_goal(goal))

        ray = ray.view(ray.size(0), -1)
        ray = self.relu(self.fc_ray(ray))

        img_goal_ray = torch.cat((depth_img, goal, ray), 1)
        img_goal_ray = self.relu(self.img_goal_ray1(img_goal_ray))
        value = self.tanh(self.img_goal_ray2(img_goal_ray))

        return value
