import torch.nn as nn
import torch


class Value(nn.Module):
    def __init__(self, HIST):
        super(Value, self).__init__()
        """ layers for inputs of depth_images """
        self.conv1 = nn.Conv2d(HIST, 32, (10, 14), (8, 8), padding=(1, 4))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), 2, padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1))
        self.fc_img = nn.Linear(8 * 10 * 64, 512)

        """ layers for inputs of goals """
        self.fc_goal = nn.Linear(HIST * 3, 96)
        self.fc_ray = nn.Linear(HIST * 1, 32)
        self.fc_action = nn.Linear(HIST * 2, 64)

        """ layers for inputs concatenated information """
        self.img_goal_ray1 = nn.Linear(704, 512)
        self.img_goal_ray2 = nn.Linear(512, 1)  # two dimensions of actions: upward and downward; turning

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.img_goal_ray2.weight.data.mul_(1)
        self.img_goal_ray2.bias.data.mul_(0.0)

    def forward(self, depth_img, goal, ray, hist_action):
        depth_img = self.relu(self.conv1(depth_img))
        depth_img = self.relu(self.conv2(depth_img))
        depth_img = self.relu(self.conv3(depth_img))
        depth_img = depth_img.view(depth_img.size(0), -1)
        depth_img = self.relu(self.fc_img(depth_img))

        goal = goal.view(goal.size(0), -1)
        goal = self.relu(self.fc_goal(goal))

        ray = ray.view(ray.size(0), -1)
        ray = self.relu(self.fc_ray(ray))

        hist_action = hist_action.view(hist_action.size(0), -1)
        hist_action = self.relu(self.fc_action(hist_action))

        img_goal_ray_aciton = torch.cat((depth_img, goal, ray, hist_action), 1)
        img_goal_ray_aciton = self.relu(self.img_goal_ray1(img_goal_ray_aciton))
        value = self.tanh(self.img_goal_ray2(img_goal_ray_aciton))

        return value
