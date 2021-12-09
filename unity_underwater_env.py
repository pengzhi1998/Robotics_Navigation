from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np
import cv2
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import uuid
import os
import glob
import torch
import time

import DPT.util.io

from torchvision.transforms import Compose

from DPT.dpt.models import DPTDepthModel
from DPT.dpt.midas_net import MidasNet_large
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

class DPT_depth():
    def __init__(self, model_type="dpt_hybrid", model_path="DPT/weights/dpt_hybrid-midas-501f0c75.pt",
                 optimize=True):
        self.optimize = optimize

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        time0 = time.time()
        img = DPT.util.io.image_input(rgb_img)
        time1 = time.time()
        img_input = self.transform({"image": img})["image"]
        time2 = time.time()

        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
            time3 = time.time()
            if self.optimize == True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                time4 = time.time()
                sample = sample.half()
                time5 = time.time()
            prediction = self.model.forward(sample)
            time6 = time.time()
            # prediction = (
            #     torch.nn.functional.interpolate(
            #         prediction.unsqueeze(1),
            #         size=img.shape[:2],
            #         mode="bicubic",
            #         align_corners=False,
            #     )
                    # .squeeze()
                    # .cpu()
                    # .numpy()
            # )

        # print("time:", time1 - time0, time2 - time1, time3 - time2, time6 - time3)
        print("time:", time1 - time0, time2 - time1, time3 - time2, time4 - time3, time5 - time4, time6 - time5)

        return prediction

class PosChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        self.goal = msg.read_float32_list()

    def goal_info(self):
        return self.goal

class Underwater_navigation():
    def __init__(self):
        self.pos_info = PosChannel()
        self.dpt = DPT_depth()
        unity_env = UnityEnvironment("/home/pengzhi1998/Unity/ml-agents/environments/water", side_channels=[self.pos_info])
        self.env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

    def reset(self):
        self.step_count = 0
        obs_img_ray = self.env.reset()
        obs_goal = self.pos_info.goal_info()
        return [obs_img_ray[0], np.min([obs_img_ray[1][1], obs_img_ray[1][3], obs_img_ray[1][5]]) * 12 * 0.8, obs_goal]


    def step(self, action):
        # action[0] controls its vertical speed, action[1] controls its rotation speed
        action_ver = action[0]/5
        action_rot = action[1] * np.pi/6
        obs_img_ray, _, done, _ = self.env.step([action_ver, action_rot])
        obs_goal = self.pos_info.goal_info()
        done = False

        pred_depth_img = self.dpt.run(obs_img_ray[0])

        # compute reward
        # 1. give a negative reward when robot is too close to nearby obstacles
        obstacle_dis = np.min([obs_img_ray[1][1], obs_img_ray[1][3], obs_img_ray[1][5],
                             obs_img_ray[1][7], obs_img_ray[1][9], obs_img_ray[1][11],
                             obs_img_ray[1][13], obs_img_ray[1][15], obs_img_ray[1][17],
                             obs_img_ray[1][19], obs_img_ray[1][21]]) * 12 * 0.8
        if obstacle_dis < 0.6:
            reward_obstacle = -10
            done = True
            print("Too close to the obstacle!\n\n\n")
        else:
            reward_obstacle = 0

        # 2. give a positive reward if the robot reaches the goal
        if obs_goal[0] < 0.3 and obs_goal[1] < 0.05:
            reward_goal_reached = 10
            done = True
            print("Reached the goal area!\n\n\n")
        else:
            reward_goal_reached = 0

        # 3. give a positive reward if the robot is reaching the goal
        reward_goal_reaching = (-np.abs(np.deg2rad(obs_goal[2])) + np.pi / 3) / 10

        # 4. give a negative reward if the robot usually turns its directions
        reward_turning = - np.abs(action_rot) / 10

        reward = reward_obstacle + reward_goal_reached + reward_goal_reaching + reward_turning
        self.step_count += 1

        if self.step_count > 500:
            done = True
            print("Exceeds the max num_step...\n\n\n")

        # print("rewards of step", self.step_count, ":", reward_obstacle, reward_goal_reached,
        #       reward_goal_reaching, reward_turning, reward)

        # the observation value for ray should be scaled
        return [obs_img_ray[0], np.min([obs_img_ray[1][1], obs_img_ray[1][3], obs_img_ray[1][5]]) * 12 * 0.8, obs_goal], \
               reward, done, 0

env = Underwater_navigation()

while True:
    done = False
    obs = env.reset()
    # cv2.imwrite("img1.png", 256 * cv2.cvtColor(obs[0], cv2.COLOR_RGB2BGR))
    while not done:
        obs, reward, done, _ = env.step([0.0, - 1.0])
        # print(obs[1], np.shape(obs[1]))
        # cv2.imwrite("img2.png", 256 * cv2.cvtColor(obs[0], cv2.COLOR_RGB2BGR))