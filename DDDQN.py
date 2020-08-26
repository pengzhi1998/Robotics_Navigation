import torch
import torch.nn as nn
import numpy as np
from GazeboWorld import GazeboWorld
import rospy
import os
import time
import random
import matplotlib.pyplot as plt
import time
from visdom import Visdom
viz = Visdom()

fig = plt.figure()
from collections import deque
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor

np.set_printoptions(threshold=np.inf)

GAME = 'GazeboWorld'
ACTIONS = 5 # number of valid actions
SPEED = 2 # DoF of speed
GAMMA = 0.97 # decay rate of past observations
OBSERVE = 1. # timesteps to observe before training
EXPLORE = 20000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
INITIAL_EPSILON = .5 # starting value of epsilon
REPLAY_MEMORY = 60000 # number of previous transitions to remember
BATCH = 8 # size of minibatch
MAX_EPISODE = 2000
MAX_T = 200
DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
RGB_IMAGE_HEIGHT = 228
RGB_IMAGE_WIDTH = 304
CHANNEL = 3
TARGET_UPDATE = 100 # every 1500 steps, we need to update the target network with the parameters in online network
H_SIZE = 8*10*64
IMAGE_HIST = 4

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

def train():
    learning_rate = 1.0e-5

    online_net = DDDQN()
    target_net = DDDQN()

    if os.path.isfile('../stored_files/online_with_noise.pth.tar') and os.path.isfile('../stored_files/target_with_noise.pth.tar'):
        resume_file_online = '../stored_files/online_with_noise.pth.tar'
        checkpoint_online = torch.load(resume_file_online)
        online_net.load_state_dict(checkpoint_online['state_dict'])
        resume_file_target = '../stored_files/target_with_noise.pth.tar'
        checkpoint_target = torch.load(resume_file_target)
        target_net.load_state_dict(checkpoint_target['state_dict'])

    online_net = online_net.cuda()
    target_net = target_net.cuda()
    # optimizer = torch.optim.Adam(online_net.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss().cuda()

    rospy.sleep(1.)

    env = GazeboWorld()
    print 'Environment initialized'

    D = deque()
    terminal = False

    episode = 0
    epsilon = INITIAL_EPSILON
    Step = 0 # the step counter to indicate whether update the network, very similar to the parameter "t", but
             # but "t" is an inside loop parameter, which means every episode the "t" will be redefined
    rate = rospy.Rate(3)
    # loop_time = time.time()
    # last_loop_time = loop_time # ?
    ten_episode_evaluation = 0
    while episode < MAX_EPISODE and not rospy.is_shutdown():
        env.ResetWorld()

        depth_img_t1 = env.GetDepthImageObservation()
        reward_t, terminal, reset, total_evaluation, goal_t1 = env.GetRewardAndTerminate(0)
        depth_imgs_t1 = np.stack((depth_img_t1, depth_img_t1, depth_img_t1, depth_img_t1), axis=0)
        goals_t1 = np.stack((goal_t1, goal_t1, goal_t1, goal_t1), axis=0)

        optimizer = torch.optim.Adam(online_net.parameters(), lr=learning_rate)
        online_net.train()
        # some settings
        t = 0
        r_epi = 0.
        terminal = False
        reset = False
        # loop_time_buf = []
        action_index = 0
        loss_sum = 0

        while not reset and not rospy.is_shutdown():
            depth_img_t1 = env.GetDepthImageObservation()
            reward_t, terminal, reset, total_evaluation, goal_t1 = env.GetRewardAndTerminate(t)
            goal_t1 = np.reshape(goal_t1, (1, 2))
            goals_t1 = np.append(goal_t1, goals_t1[:(IMAGE_HIST - 1), :], axis=0)
            depth_img_t1 = np.reshape(depth_img_t1, (1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH))
            depth_imgs_t1 = np.append(depth_img_t1, depth_imgs_t1[:(IMAGE_HIST - 1), :, :], axis=0)

            if reset == True:
                ten_episode_evaluation += total_evaluation # to compute the average reward over 50 episodes
            if t > 0:
                # depth_imgs_t is the state images for former time, depth_imgs_t1 is the state images for latter time
                print depth_imgs_t1.shape, goals_t1.shape
                D.append((depth_imgs_t, a_t, reward_t, depth_imgs_t1, terminal, goals_t, goals_t1))
                if len(D) > REPLAY_MEMORY:
                    D.popleft()
            depth_imgs_t = depth_imgs_t1
            depth_imgs_t1_cuda = depth_imgs_t1[np.newaxis, :]
            depth_imgs_t1_cuda = torch.from_numpy(depth_imgs_t1_cuda)
            depth_imgs_t1_cuda = Variable(depth_imgs_t1_cuda.type(dtype))
            goals_t = goals_t1
            goals_t1_cuda = goals_t1[np.newaxis, :]
            goals_t1_cuda = torch.from_numpy(goals_t1_cuda)
            goals_t1_cuda = Variable(goals_t1_cuda.type(dtype))

            a = online_net(depth_imgs_t1_cuda, goals_t1_cuda) # the shape of a is [1, 5] which has two dimensions, so we need a[0] to have a dimensionality reduction
            # readout_t = a[0].cpu().detach().numpy()
            readout_t = a[0]
            a_t = np.zeros([ACTIONS]) # ([0., 0., 0., 0., 0.])

            # there aren't enough data in the buffer, so if the episode is not big enough, we just collect them
            if episode <= 10:
                action_index = random.randrange(ACTIONS) # any number from 0-5
                a_t[action_index] = 1 # let the specified action value to be 1
            else: # the data is enough, so let's begin to train the network to take some reasonable steps
                rdnum = random.random()
                # if rdnum <= epsilon: # choose a random action
                if rdnum <= epsilon:
                    print "-------------Random Action---------------"
                    action_index = random.randrange(ACTIONS)
                    a_t[action_index] = 1
                else:
                    max_q_value, action_index = torch.max(readout_t, 0)
                    a_t[action_index] = 1
                # control the agent
            env.Control(action_index)
            # print "action"
            # at the same time, we need to train the network
            if episode > OBSERVE:
                minibatch = random.sample(D, BATCH)
                y_batch = []
                # get the batch variables
                depth_imgs_t_batch = torch.FloatTensor([d[0] for d in minibatch]) # the former batch
                depth_imgs_t_batch = Variable(depth_imgs_t_batch.type(dtype))
                a_batch = torch.FloatTensor([d[1] for d in minibatch])
                a_batch = Variable(a_batch.type(dtype))
                r_batch = torch.FloatTensor([d[2] for d in minibatch])
                r_batch = Variable(r_batch.type(dtype))
                depth_imgs_t1_batch = torch.FloatTensor([d[3] for d in minibatch]) # the latter batch
                depth_imgs_t1_batch = Variable(depth_imgs_t1_batch.type(dtype))
                goals_t_batch = torch.FloatTensor([d[5] for d in minibatch])
                goals_t_batch = Variable(goals_t_batch.type(dtype))
                goals_t1_batch = torch.FloatTensor([d[6] for d in minibatch])
                goals_t1_batch = Variable(goals_t1_batch.type(dtype))

                # print depth_imgs_t1_batch.shape
                Q1 = online_net(depth_imgs_t1_batch, goals_t1_batch)
                Q2 = target_net(depth_imgs_t1_batch, goals_t1_batch)
                for i in range(0, len(minibatch)):
                    terminal_batch = minibatch[i][4]
                    if terminal_batch:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * Q2[i, torch.argmax(Q1[i])])
                # the culculation with Q(s', a')
                y_batch = torch.FloatTensor(y_batch)
                y_batch = Variable(y_batch.type(dtype))
                # Q(s, a)
                Q_current = online_net(depth_imgs_t_batch, goals_t_batch)
                Q_predicted_value = torch.sum(torch.mul(Q_current, a_batch), 1)

                loss = loss_func(y_batch, Q_predicted_value)
                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()

            # update the network every 100 steps
            if (Step + 1) % TARGET_UPDATE == 0:
                target_net.load_state_dict(online_net.state_dict())
            Step += 1
            print "episode:", episode, "\nStep:", Step

            r_epi = r_epi + reward_t
            t += 1
            rate.sleep()

            # scale down epsilon
            if epsilon > FINAL_EPSILON and episode > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # save the network every 500 episodes
        if (episode + 1) % 150 == 0:
            torch.save({
                'episode': episode + 1,
                'state_dict': online_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epsilon': episode
            }, '../stored_files/online_with_noise.pth.tar')
            learning_rate = learning_rate * 0.96
        if (episode + 51) % 150 == 0:
            torch.save({
                'episode': episode + 1,
                'state_dict': online_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epsilon': episode
            }, '../stored_files/target_with_noise.pth.tar')

        print "episode:", episode, ", loss:", loss_sum / t, ", total reward for this episode:", total_evaluation

        if (episode + 1) % 10 == 0:
            average_evaluation = ten_episode_evaluation / 10 # type: int
            print "the average reward evaluation:", average_evaluation
            viz.line(
                Y = np.expand_dims(np.array(average_evaluation), axis = 0),
                X = np.expand_dims(np.array(episode), axis = 0),
                win = 'reward',
                update='append'
            )
            ten_episode_evaluation = 0
            my_open = open("../stored_files/data.txt", "a")
            data = [str(episode), str(average_evaluation), "\n"]
            for element in data:
                my_open.write(element)
            my_open.close()

        episode += 1

def main():
    train()

if __name__ == "__main__":
    main()
