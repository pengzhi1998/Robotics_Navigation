import rospy
import torch
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import tf
from cv_bridge import CvBridge, CvBridgeError
batch_size = 1
from torch.autograd import Variable
import torch.nn as nn
dtype = torch.cuda.FloatTensor
import cv2
import numpy as np
IMAGE_HIST = 4
import argparse
import time
import torchvision.transforms as transforms
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from models.models import create_model
model = create_model(opt)
model.switch_to_eval()

class UWsimWorld():
    def __init__(self):
        # self.goal_table = np.array([[-20, -5], [-12, 9], [0, 0], [-10, -14], [-19, 8]]) # goals for multi..
        # self.goal_table = np.array([[-12, -8]]) # goal for L (single) obstacle
        # self.goal_table = np.array([[-12, -7.5]])  # goal for channel
        self.goal_table = np.array([[-15, 17]])  # goal for shipwreck obstacle
        rospy.init_node('uwsim', anonymous=False)
        self.cmd_vel = rospy.Publisher('g500/velocityCommand', TwistStamped, queue_size=10)
        # self.cmd_vel = rospy.Publisher('vel', TwistStamped, queue_size=10)
        self.gps = rospy.Subscriber('g500/gps', NavSatFix, self.GetGPS)
        self.orientation = rospy.Subscriber('g500/imu', Imu, self.GetOri)
        self.rgb_image_sub = rospy.Subscriber('g500/camera1', Image, self.RGBImageCallBack)
        # self.rgb_image_sub = rospy.Subscriber('camera/image_raw/compressed', CompressedImage, self.RGBImageCallBack)
        self.echo_sounder_sub = rospy.Subscriber('/g500/multibeam', LaserScan, self.MultibeamCallBack)
        self.action_table = [-np.pi / 12, -np.pi / 24, 0., np.pi / 24, np.pi / 12]
        self.depth_image_size = [160, 128]
        self.rgb_image_size = [512, 384]
        self.bridge = CvBridge()
        self.i = 0

    def GetGPS(self, gps):
        self.cur_pos = np.array([-gps.latitude, gps.longitude])
        self.R2G = self.goal - self.cur_pos

    def GetPosition(self):
        return(self.cur_pos, self.goal)

    def choose_goal(self):
        self.goal = self.goal_table[self.i]
        self.i = self.i + 1

    def GetOri(self, orientation):
        quaternion = (orientation.orientation.x, orientation.orientation.y, orientation.orientation.z,
                          orientation.orientation.w)
        self.euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = - self.euler[2]
        if yaw < 0:
            yaw = yaw + 2 * np.pi
        self.rob_ori = np.array([np.cos(yaw), np.sin(yaw)])

    def Goal(self):
        distance = np.sqrt(self.R2G[0] ** 2 + self.R2G[1] ** 2)/4

        angle = np.arccos(self.R2G.dot(self.rob_ori) / np.sqrt((self.R2G.dot(self.R2G)) * np.sqrt(self.rob_ori.dot(self.rob_ori))))*1.5

        # determine whether the goal is on the right or left hand side of the robot
        if self.rob_ori[0] > 0 and (self.rob_ori[1] / self.rob_ori[0]) * self.R2G[0] > self.R2G[1]:
            angle = -angle
        elif self.rob_ori[0] < 0 and (self.rob_ori[1] / self.rob_ori[0]) * self.R2G[0] < self.R2G[1]:
            angle = -angle
        elif self.rob_ori[0] == 0:
            if self.rob_ori[1] > 0 and self.R2G[0] > 0:
                angle = -angle
            elif self.rob_ori[1] < 0 and self.R2G[0] < 0:
                angle = -angle

        goal = np.array([distance, angle])
        print("goal:", goal)
        return(goal)

    def RGBImageCallBack(self, img):
        self.rgb_image = img

    def MultibeamCallBack(self, multibeam):
        self.multibeam = multibeam

    def Multibeam(self):
        multibeam = self.multibeam.ranges
        return multibeam

    def GetRGBImageObservation(self):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
            # cv_img = self.bridge.compressed_imgmsg_to_cv2(self.rgb_image, "bgr8")
        except Exception as e:
            raise e
        # resize
        cv2.imwrite("img.png", cv_img)
        dim = (self.rgb_image_size[0], self.rgb_image_size[1])
        cv_img = np.float32(cv_img)/255
        cv_resized_img = cv2.resize(cv_img, dim, interpolation=cv2.INTER_AREA)
        cv_resized_img = cv2.cvtColor(cv_resized_img, cv2.COLOR_BGR2RGB)
        # cv2 image to ros image and publish
        # try:
        #     resized_img = self.bridge.cv2_to_imgmsg(cv_resized_img, "bgr8")
        # except Exception as e:
        #     raise e
        return (cv_resized_img)

    def Control(self, forward, twist):
        # if action < 2:
        # 	self.self_speed[0] = self.action_table[action]
        # 	# self.self_speed[1] = 0.
        # else:
        # 	self.self_speed[1] = self.action_table[action]
        move_cmd = TwistStamped()
        move_cmd.twist.linear.x = forward
        move_cmd.twist.linear.y = 0.
        move_cmd.twist.linear.z = 0.
        move_cmd.twist.angular.x = 0.
        move_cmd.twist.angular.y = 0.
        move_cmd.twist.angular.z = self.action_table[twist]
        self.motion = move_cmd
        self.cmd_vel.publish(move_cmd)



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

def DRL():
    time_start = time.time()
    bridge = CvBridge()
    depth_img_pub = rospy.Publisher('depth', Image, queue_size=10)
    # define and load depth_prediction network

    # define and load q_learning network
    online_net = DDDQN()
    # resume_file_online = '../DDDQN_goal_ppo/online_with_noise.pth.tar'
    # resume_file_online = '../DDDQN_goal_ppo/online_with_noise_32_shaped.pth.tar'
    # resume_file_online = '../DDDQN_goal_ppo/online_with_noise_32_unshaped.pth.tar'
    resume_file_online = '../DDDQN_goal_ppo/online_with_noise_64_small_shaped.pth.tar'
    checkpoint_online = torch.load(resume_file_online)
    online_net.load_state_dict(checkpoint_online['state_dict'])
    episode_number = checkpoint_online['episode']
    online_net.cuda()
    rospy.sleep(1.)

    # Initialize the World and variables
    env = UWsimWorld()
    env.choose_goal()
    print('Environment initialized')
    rospy.sleep(2)

    # start training
    rate = rospy.Rate(3)
    with torch.no_grad():
        while not rospy.is_shutdown():
            t = 0
            rgb_img_t1 = env.GetRGBImageObservation()
            goal_t1 = env.Goal()

            input_img = torch.from_numpy(np.transpose(rgb_img_t1, (2, 0, 1))).contiguous().float()
            input_img = input_img.unsqueeze(0)
            input_img = Variable(input_img.cuda())
            pred_log_depth = model.netG.forward(input_img)
            # pred_log_depth = torch.squeeze(pred_log_depth)

            depth_img_t1 = torch.exp(pred_log_depth)
            depth_img_cpu = depth_img_t1[0].data.squeeze().cpu().numpy().astype(np.float32)
            depth_img_cpu = bridge.cv2_to_imgmsg(depth_img_cpu, "passthrough")
            depth_img_pub.publish(depth_img_cpu)
            # depth_img_t1 *= 3
            # depth_img_t1[depth_img_t1 >= 3.] = 3.  # np.random.uniform(.8, 1.2)
            # depth_img_t1 += 2.
            # depth_img_t1[depth_img_t1 <= 3] = 1.5
            # depth_img_t1[depth_img_t1 < 1.8] -= .5
            # depth_img_t1[depth_img_t1 > 2.] = 3.5
            depth_img_t1 *= 5
            depth_img_t1[depth_img_t1 <= 2] -= .5
            depth_img_t1[depth_img_t1 >= 5.] = 0.0
            depth_img_t1 = nn.functional.interpolate(depth_img_t1, size=(128, 160))
            depth_img_t1 = torch.squeeze(depth_img_t1, 1)
            depth_imgs_t1 = torch.stack((depth_img_t1, depth_img_t1, depth_img_t1, depth_img_t1), dim=1)
            goals_t1 = np.stack((goal_t1, goal_t1, goal_t1, goal_t1), axis=0)

            while not rospy.is_shutdown():
                multibeam = env.Multibeam()
                rgb_img_t1 = env.GetRGBImageObservation()
                goal_t1 = env.Goal()
                if goal_t1[0] < .25:
                    env.choose_goal()
                    # rospy.signal_shutdown("stopped")

                # rgb_img_t1 = np.float32(cv2.imread("../pool_images/Screenshot from 2020-10-23 21-29-45.png"))/255
                # rgb_img_t1 = np.float32(cv2.imread("under_water_depth.png")) / 255
                # rgb_img_t1 = np.float32(cv2.imread("../lake_images/1547660794.385960.png"))/255
                rgb_img_t1 = cv2.resize(rgb_img_t1, (512, 384), interpolation=cv2.INTER_LINEAR)
                rgb_img_t1 = cv2.cvtColor(rgb_img_t1, cv2.COLOR_BGR2RGB)

                input_img = torch.from_numpy(np.transpose(rgb_img_t1, (2, 0, 1))).contiguous().float()
                input_img = input_img.unsqueeze(0)
                input_img = Variable(input_img.cuda())

                goal = goal_t1
                pred_log_depth = model.netG.forward(input_img)
                # pred_log_depth = torch.squeeze(pred_log_depth)

                depth_img_t1 = torch.exp(pred_log_depth)

                depth_img_t1 = nn.functional.interpolate(depth_img_t1, size=(128, 160))


                # depth_img_t1[depth_img_t1 <= 3] -= 1.5
                # depth_img_t1[depth_img_t1 == 5] = 0
                # depth_img_t1 += 2.
                # depth_img_t1[depth_img_t1 <= 2] -= .5
                # depth_img_t1[depth_img_t1 <= 0] = 0
                print torch.min(depth_img_t1), torch.max(depth_img_t1)
                depth_img = depth_img_t1.permute(0, 2, 3, 1)
                depth_img = depth_img[0].data.squeeze().cpu().numpy().astype(np.float32)
                # cv2.imwrite("under_water_depth.png", depth_img * 50)

                # the scale for the new model (with new reward and 1500 episodes training,
                # online_with_noise_64_small_shaped.pth.tar)
                depth_img_t1 *= 3
                depth_img_t1[depth_img_t1 >= 3.] = 3.  # np.random.uniform(.8, 1.2)
                depth_img_t1 += 2.
                # the scale for the old one (with old reward and 900 episodes training,
                # online_with_noise.pth.tar
                # depth_img_t1 *= 5
                # depth_img_t1[depth_img_t1 <= 2] -= .5
                # depth_img_t1[depth_img_t1 >= 5.] = 0.0  # np.random.uniform(.8, 1.2)
                print episode_number
                # depth_img_t1 *= 2
                # depth_img_t1 += 4
                # depth_img_t1[depth_img_t1 <= 2] -= .5
                # depth_img_t1[depth_img_t1 >= 5.] = 0.0  # np.random.uniform(.8, 1.2)
                # depth_img_t1[depth_img_t1 == 0] = -0.5
                # depth_img_t1 *= 2
                # depth_img_t1 += 1

                depth_img_cpu = depth_img_t1.permute(0, 2, 3, 1)
                depth_img_cpu = depth_img_cpu[0].data.squeeze().cpu().numpy().astype(np.float32)
                depth_img_cpu = bridge.cv2_to_imgmsg(depth_img_cpu, "passthrough")
                depth_img_pub.publish(depth_img_cpu)

                depth_imgs_t1 = torch.cat((depth_img_t1, depth_imgs_t1[:, :(IMAGE_HIST - 1), :, :]), 1)
                depth_imgs_t1_cuda = Variable(depth_imgs_t1.type(dtype))
                goal_t1 = np.reshape(goal_t1, (1, 2))
                goals_t1 = np.append(goal_t1, goals_t1[:(IMAGE_HIST - 1), :], axis=0)
                goals_t1_cuda = goals_t1[np.newaxis, :]
                goals_t1_cuda = torch.from_numpy(goals_t1_cuda)
                goals_t1_cuda = Variable(goals_t1_cuda.type(dtype))

                Q_value_list = online_net(depth_imgs_t1_cuda, goals_t1_cuda)
                Q_value_list = Q_value_list[0]
                print Q_value_list
                Q_value, action = torch.max(Q_value_list, 0)
                # env.Control(0.25, action)
                if multibeam[50] > 1.2 and multibeam[40] > 1.2 and multibeam[60] > 1.2:
                    print multibeam[50], multibeam[40], multibeam[60]
                    env.Control(0.25, action)
                else:
                    a = 0
                    while a < 40000:
                        a += 1
                        env.Control(0, 4)
                    distance_right = (env.Multibeam())[50]
                    a = 0
                    while a < 80000:
                        a += 1
                        env.Control(0, 0)
                    distance_left = (env.Multibeam())[50]
                    # if multibeam[20] > multibeam[80]: # right handside has bigger area
                    if distance_right > distance_left:
                        a = 0
                        while a < 100000:
                            a += 1
                            env.Control(0, 4) # turn right
                    else:
                        a = 0
                        while a < 20000:
                            a += 1
                            env.Control(0, 0)  # turn left

                if t % 5 == 0:
                    path_file = open("path_shipwreck_megaDRL.txt", "a")
                    cur_pos, goal_pos = env.GetPosition()
                    data = [str(cur_pos[0]), ' ', str(cur_pos[1]), "\n"]
                    # for element in data:
                    #     path_file.write(element)
                    path_file.close()

                t += 1
                rate.sleep()
                if(goal[0]<1):
                    time_end = time.time()
                    print "spend time:", time_end - time_start,'s'

def main():
    DRL()


if __name__ == "__main__":
    main()
