import torch
from GazeboWorld import GazeboWorld
import torch.nn as nn
import numpy as np
import rospy
DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
IMAGE_HIST = 4
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor

env = GazeboWorld()

class DDDQN(nn.Module):
    def __init__(self):
        super(DDDQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, (10, 14), (8, 8), padding = (1, 4))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), 2, padding = (1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1, padding = (1, 1))

        self.img_goal_adv1 = nn.Linear(8*10*64, 512)
        self.img_goal_val1 = nn.Linear(8*10*64, 512)
        self.fc_goal = nn.Linear(4 * 2, 64)

        self.img_goal_adv2 = nn.Linear(576, 512)
        self.img_goal_val2 = nn.Linear(576, 512)
        self.img_goal_adv3 = nn.Linear(512, 5)
        self.img_goal_val3 = nn.Linear(512, 1)

        self.relu = nn.ReLU()

    def forward(self, x, goal): # remember to initialize
        # batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        adv = self.relu(self.img_goal_adv1(x))
        val = self.relu(self.img_goal_val1(x))

        goal = goal.view(goal.size(0), -1)
        goal = self.relu(self.fc_goal(goal))

        adv = torch.cat((adv, goal), 1) # concatenate the feature map of the image as well as
        val = torch.cat((val, goal), 1)
        # the information of the goal position
        adv = self.relu(self.img_goal_adv2(adv))
        val = self.relu(self.img_goal_val2(val))

        adv = self.img_goal_adv3(adv)
        val = self.img_goal_val3(val).expand(x.size(0), 5) # shape = [batch_size, 5]
        # print adv.mean(1).unsqueeze(1).shape
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), 5) # shape = [batch_size, 5]
        # print x.shape
        return x

test_net = DDDQN()
resume_file = '../stored_files/online_with_noise.pth.tar'
checkpoint = torch.load(resume_file)
test_net.load_state_dict(checkpoint['state_dict'])
test_net.cuda()

def test():
    while True:
        env.ResetWorld()
        depth_img_t1 = env.GetDepthImageObservation()
        reward_t, terminal, reset, total_evaluation, goal_t1 = env.GetRewardAndTerminate(0)
        depth_imgs_t1 = np.stack((depth_img_t1, depth_img_t1, depth_img_t1, depth_img_t1), axis=0)
        goals_t1 = np.stack((goal_t1, goal_t1, goal_t1, goal_t1), axis=0)
        rate = rospy.Rate(3)
        rospy.sleep(1.)
        test_net.eval()
        with torch.no_grad():
            testing_loss = 0
            reset = False
            t = 0
            while not reset and not rospy.is_shutdown():
                depth_img_t1 = env.GetDepthImageObservation()
                reward_t, terminal, reset, total_evaluation, goal_t1 = env.GetRewardAndTerminate(t)
                # print depth_img_t1.max()
                depth_img_t1 = np.reshape(depth_img_t1, (1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH))
                depth_imgs_t1 = np.append(depth_img_t1, depth_imgs_t1[:(IMAGE_HIST - 1), :, :], axis=0)
                depth_imgs_t1_cuda = depth_imgs_t1[np.newaxis, :]
                depth_imgs_t1_cuda = torch.from_numpy(depth_imgs_t1_cuda)
                depth_imgs_t1_cuda = Variable(depth_imgs_t1_cuda.type(dtype))
                goal_t1 = np.reshape(goal_t1, (1, 2))
                goals_t1 = np.append(goal_t1, goals_t1[:(IMAGE_HIST - 1), :], axis=0)
                goals_t1_cuda = goals_t1[np.newaxis, :]
                goals_t1_cuda = torch.from_numpy(goals_t1_cuda)
                goals_t1_cuda = Variable(goals_t1_cuda.type(dtype))

                Q_value_list = test_net(depth_imgs_t1_cuda, goals_t1_cuda)
                # print Q_value_list, reset
                Q_value_list = Q_value_list[0]
                Q_value, action = torch.max(Q_value_list, 0)
                env.Control(action)
                t += 1
                rate.sleep()

def main():
    test()

if __name__ == "__main__":
    main()