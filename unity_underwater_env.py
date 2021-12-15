import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import time
import uuid
import os

from gym import spaces
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from torchvision.transforms import Compose
from DPT.dpt.models import DPTDepthModel
from DPT.dpt.midas_net import MidasNet_large
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
DIM_GOAL = 3
HIST = 4
BITS = 2

class DPT_depth():
    def __init__(self, device, model_type="dpt_large", model_path=
    os.path.abspath("./") + "/DPT/weights/dpt_large-midas-2f21e586.pt",
                 optimize=True):
        self.optimize = optimize
        self.THRESHOLD = torch.tensor(np.finfo("float").eps).to(device)

        # load network
        if model_type == "dpt_large":  # DPT-Large
            self.net_w = self.net_h = 384
            self.model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif model_type == "dpt_hybrid":  # DPT-Hybrid
            self.net_w = self.net_h = 384
            self.model = DPTDepthModel(
                path=model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "dpt_hybrid_kitti":
            self.net_w = 1216
            self.net_h = 352

            self.model = DPTDepthModel(
                path=model_path,
                scale=0.00006016,
                shift=0.00579,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )

            self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif model_type == "dpt_hybrid_nyu":
            self.net_w = 640
            self.net_h = 480

            self.model = DPTDepthModel(
                path=model_path,
                scale=0.000305,
                shift=0.1378,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )

            self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif model_type == "midas_v21":  # Convolutional model
            self.net_w = self.net_h = 384

            self.model = MidasNet_large(model_path, non_negative=True)
            self.normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        else:
            assert (
                False
            ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

        # select device
        self.device = device

        self.transform = Compose(
            [
                Resize(
                    self.net_w,
                    self.net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                self.normalization,
                PrepareForNet(),
            ]
        )

        self.model.eval()

        if optimize == True and self.device == torch.device("cuda"):
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model = self.model.half()

        self.model.to(self.device)

    def run(self, rgb_img):
        img_input = self.transform({"image": rgb_img})["image"]
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
            if self.optimize == True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(0),
                    size=(DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            depth_min = prediction.min()
            depth_max = prediction.max()
            if depth_max - depth_min > self.THRESHOLD:
                prediction = (prediction - depth_min) / (depth_max - depth_min)
            else:
                prediction = np.zeros(prediction.shape, dtype=prediction.dtype)

            # prediction_show = prediction.squeeze().cpu().numpy()
            # print("prediction size:", np.shape(prediction_show), np.shape(prediction))
            # plt.imshow(np.uint16(prediction_show * 65536))
            # plt.show()

            # cv2.imwrite("depth.png", ((prediction_show*65536).astype("uint16")), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return prediction

class PosChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        self.goal_depthfromwater = msg.read_float32_list()

    def goal_depthfromwater_info(self):
        return self.goal_depthfromwater

class Underwater_navigation():
    def __init__(self):
        self.twist_range = 30 # degree
        self.vertical_range = 0.1
        self.action_space = spaces.Box(
            np.array([-self.twist_range, -self.vertical_range]).astype(np.float32),
            np.array([self.twist_range, self.vertical_range]).astype(np.float32),
        )
        self.observation_space_img_depth = (HIST, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH)
        self.observation_space_goal = (HIST, DIM_GOAL)
        self.observation_space_ray = (HIST, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_info = PosChannel()
        config_channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(os.path.abspath("./") + "/underwater_env/water",
                                     side_channels=[config_channel, self.pos_info])
        config_channel.set_configuration_parameters(time_scale=1, capture_frame_rate=100)
        self.env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

        self.dpt = DPT_depth(self.device)

    def reset(self):
        self.step_count = 0
        obs_img_ray = self.env.reset()

        # observations per frame
        obs_preddepth = 1 - self.dpt.run(obs_img_ray[0])
        obs_ray = np.array([np.min([obs_img_ray[1][1], obs_img_ray[1][3], obs_img_ray[1][5]]) * 12 * 0.8])
        obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())

        # construct the observations of depth images, goal infos, and rays for consecutive 4 frames
        self.obs_preddepths = np.stack((obs_preddepth, obs_preddepth, obs_preddepth, obs_preddepth),
                                        axis=0) # torch.Size([1, 4, 128, 160])

        self.obs_goals = np.stack((obs_goal_depthfromwater[:3], obs_goal_depthfromwater[:3],
                                   obs_goal_depthfromwater[:3], obs_goal_depthfromwater[:3]), axis=0)

        self.obs_rays = np.stack((obs_ray, obs_ray, obs_ray, obs_ray), axis=0)

        return self.obs_preddepths, self.obs_goals, self.obs_rays

    def step(self, action):
        self.time_before = time.time()
        # action[0] controls its vertical speed, action[1] controls its rotation speed
        action_ver = action[0]
        action_rot = action[1] * self.twist_range

        # observations per frame
        obs_img_ray, _, done, _ = self.env.step([action_ver, action_rot])
        time0 = time.time()
        obs_preddepth = self.dpt.run(obs_img_ray[0])
        time1 = time.time()
        # print("time:", time1 - time0)
        obs_ray = np.min([obs_img_ray[1][1], obs_img_ray[1][3], obs_img_ray[1][5]]) * 12 * 0.8
        obs_goal_depthfromwater = self.pos_info.goal_depthfromwater_info()

        # compute reward
        # 1. give a negative reward when robot is too close to nearby obstacles, seafloor or the water surface
        obstacle_distance = np.min([obs_img_ray[1][1], obs_img_ray[1][3], obs_img_ray[1][5],
                             obs_img_ray[1][7], obs_img_ray[1][9], obs_img_ray[1][11],
                             obs_img_ray[1][13], obs_img_ray[1][15], obs_img_ray[1][17],
                             obs_img_ray[1][19], obs_img_ray[1][21]]) * 12 * 0.8
        if obstacle_distance < 0.6 or np.abs(obs_goal_depthfromwater[3]) < 0.3\
                or np.abs(obs_goal_depthfromwater[3]+3) < 0.3:
            reward_obstacle = -10
            done = True
            print("Too close to the obstacle, seafloor or water surface!\n\n\n")
        else:
            reward_obstacle = 0

        # 2. give a positive reward if the robot reaches the goal
        if obs_goal_depthfromwater[0] < 0.4 and np.abs(obs_goal_depthfromwater[1]) <0.2:
            reward_goal_reached = 10
            done = True
            print("Reached the goal area!\n\n\n")
        else:
            reward_goal_reached = 0

        # 3. give a positive reward if the robot is reaching the goal
        reward_goal_reaching = (-np.abs(np.deg2rad(obs_goal_depthfromwater[2])) + np.pi / 3) / 10
        if (obs_goal_depthfromwater[1] > 0 and action_ver > 0) or\
                (obs_goal_depthfromwater[1] < 0 and action_ver < 0):
            reward_goal_reaching += 0.05
            # print("reaching the goal vertically", obs_goal_depthfromwater[1], action_ver)
        else:
            reward_goal_reaching -= 0.05
            # print("being away from the goal vertically", obs_goal_depthfromwater[1], action_ver)

        # 4. give a negative reward if the robot usually turns its directions
        reward_turning = - np.abs(action_rot) / 600

        reward = reward_obstacle + reward_goal_reached + reward_goal_reaching + reward_turning
        self.step_count += 1

        if self.step_count > 500:
            done = True
            print("Exceeds the max num_step...\n\n\n")

        # construct the observations of depth images, goal infos, and rays for consecutive 4 frames
        obs_preddepth = np.reshape(obs_preddepth, (1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH))
        self.obs_preddepths = np.append(obs_preddepth, self.obs_preddepths[:(HIST - 1), :, :], axis=0)

        obs_goal = np.reshape(np.array(obs_goal_depthfromwater[0:3]), (1, DIM_GOAL))
        self.obs_goals = np.append(obs_goal, self.obs_goals[:(HIST - 1), :], axis=0)

        obs_ray = np.reshape(np.array(obs_ray), (1, 1))  # single beam sonar
        self.obs_rays = np.append(obs_ray, self.obs_rays[:(HIST - 1), :], axis=0)
        self.time_after = time.time()
        print("execution_time:", self.time_after - self.time_before)
        return self.obs_preddepths, self.obs_goals, self.obs_rays, reward, done, 0
#
# env = Underwater_navigation()
#
# while True:
#     a = 0
#     done = False
#     obs = env.reset()
#     # cv2.imwrite("img1.png", 256 * cv2.cvtColor(obs[0], cv2.COLOR_RGB2BGR))
#     while not done:
#         cam, goal, ray, reward, done, _ = env.step([1.0, 0.0])
#         print(a)
#         a += 1
        # print(obs[1], np.shape(obs[1]))
        # cv2.imwrite("img2.png", 256 * cv2.cvtColor(obs[0], cv2.COLOR_RGB2BGR))
