import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import time
import uuid
import random
import os
from utils import *

from typing import List
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
from DPT.dpt.midas_net_custom import MidasNet_small
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
DIM_GOAL = 3
DIM_ACTION = 2
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
            resize_mode = "minimal"
            self.model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif model_type == "dpt_hybrid":  # DPT-Hybrid
            self.net_w = self.net_h = 384
            resize_mode = "minimal"
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
            resize_mode = "minimal"

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
            resize_mode = "minimal"

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

            resize_mode = "upper_bound"
            self.model = MidasNet_large(model_path, non_negative=True)
            self.normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        elif model_type == "midas_v21_small":
            self.net_w = self.net_h = 256
            resize_mode = "upper_bound"
            self.model = MidasNet_small(model_path, non_negative=True)

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
                    resize_method=resize_mode,
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

            # plt.imshow(np.uint16(prediction * 65536))
            # plt.show()

        # cv2.imwrite("depth.png", ((prediction*65536).astype("uint16")), [cv2.IMWRITE_PNG_COMPRESSION, 0])
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

    def assign_testpos_visibility(self, data: List[float]) -> None:
        msg = OutgoingMessage()
        msg.write_float32_list(data)
        super().queue_message_to_send(msg)

class Underwater_navigation_Bug2():
    def __init__(self, depth_prediction_model, adaptation, randomization, rank, HIST, start_goal_pos=None, training=True):
        if adaptation and not randomization:
            raise Exception("Adaptation should be used with domain randomization during training")
        self.adaptation = adaptation
        self.randomization = randomization
        self.HIST = HIST
        self.training = training
        self.twist_range = 30 # degree
        self.vertical_range = 0.1
        self.action_space = spaces.Box(
            np.array([-self.twist_range, -self.vertical_range]).astype(np.float32),
            np.array([self.twist_range, self.vertical_range]).astype(np.float32),
        )
        self.observation_space_img_depth = (self.HIST, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH)
        self.observation_space_goal = (self.HIST, DIM_GOAL)
        self.observation_space_ray = (self.HIST, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_info = PosChannel()
        config_channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(os.path.abspath("./") + "/underwater_env/test0_bug2",
                                     side_channels=[config_channel, self.pos_info], worker_id=rank, base_port=5005)

        if self.randomization == True:
            if self.training == False:
                visibility = 3 * (13 ** random.uniform(0, 1))
                if start_goal_pos == None:
                    raise AssertionError
                self.start_goal_pos = start_goal_pos
                self.pos_info.assign_testpos_visibility(self.start_goal_pos + [visibility])
        else:
            if self.training == False:
                if start_goal_pos == None:
                    raise AssertionError
                self.start_goal_pos = start_goal_pos
                visibility = 3 * (13 ** 0.5)
                self.pos_info.assign_testpos_visibility(self.start_goal_pos + [visibility])

        config_channel.set_configuration_parameters(time_scale=10, capture_frame_rate=100)
        self.env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

        if depth_prediction_model == "dpt":
            self.dpt = DPT_depth(self.device, model_type="dpt_large", model_path=
            os.path.abspath("./") + "/DPT/weights/dpt_large-midas-2f21e586.pt")
        elif depth_prediction_model == "midas":
            self.dpt = DPT_depth(self.device, model_type="midas_v21_small", model_path=
            os.path.abspath("./") + "/DPT/weights/midas_v21_small-70d6b9c8.pt")

    def reset(self):
        self.waypoint = 0
        self.step_count = 0
        self.pos_info.assign_testpos_visibility(self.start_goal_pos[0:9] + [20])

        # waiting for the initialization
        self.env.reset()
        obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
        if self.training == False:
            my_open = open(os.path.join(assets_dir(), 'learned_models/test_pos.txt'), "a")
            data = [str(obs_goal_depthfromwater[4]), " ", str(obs_goal_depthfromwater[5]), " ",
                    str(obs_goal_depthfromwater[3]), "\n"]
            for element in data:
                my_open.write(element)
            my_open.close()
        obs_img_ray, _, done, _ = self.env.step([0, 0, 0])
        self.obs_img_ray = obs_img_ray

        # observations per frame
        obs_preddepth = self.dpt.run(obs_img_ray[0] ** 0.45)
        # obs_ray = np.array([np.min([obs_img_ray[1][1], obs_img_ray[1][3], obs_img_ray[1][5],
        #                             obs_img_ray[1][33], obs_img_ray[1][35]]) * 8 * 0.5])
        obs_ray = np.array([0])
        obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())

        if self.training == False:
            my_open = open(os.path.join(assets_dir(), 'learned_models/test_pos.txt'), "a")
            data = [str(obs_goal_depthfromwater[4]), " ", str(obs_goal_depthfromwater[5]), " ",
                    str(obs_goal_depthfromwater[3]), "\n"]
            for element in data:
                my_open.write(element)
            my_open.close()

        # construct the observations of depth images, goal infos, and rays for consecutive 4 frames
        # print(np.shape(obs_preddepth), np.shape(obs_goal_depthfromwater[:3]), np.shape(obs_ray), "\n\n\n")
        self.obs_preddepths = np.array([obs_preddepth.tolist()] * self.HIST)
        self.obs_goals = np.array([obs_goal_depthfromwater[:3].tolist()] * self.HIST)
        self.obs_rays = np.array([obs_ray.tolist()] * self.HIST)
        self.obs_actions = np.array([[0, 0]] * self.HIST)
        self.done = False
        # 0 turning to the goal, 1 moving towards the goal, 2 when near the obstacle, turn to a
        self.mode = 0

        # cv2.imwrite("img_rgb_reset.png", 256 * cv2.cvtColor(obs_img_ray[0] ** 0.45, cv2.COLOR_RGB2BGR))
        # cv2.imwrite("img_depth_pred_reset.png", 256 * self.obs_preddepths[0])

        return self.obs_preddepths, self.obs_goals, self.obs_rays, self.obs_actions

    def step(self, action):
        obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
        distance = obs_goal_depthfromwater[0]
        angle = obs_goal_depthfromwater[2]
        while distance > 1:
            obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
            distance = obs_goal_depthfromwater[0]
            angle = obs_goal_depthfromwater[2]
            while angle > 10:
                obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                distance = obs_goal_depthfromwater[0]
                angle = obs_goal_depthfromwater[2]
                self.obs_img_ray, _, done, _ = self.env.step([0, 0.2 * self.twist_range, 0])
            while angle < -10:
                obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                distance = obs_goal_depthfromwater[0]
                angle = obs_goal_depthfromwater[2]
                self.obs_img_ray, _, done, _ = self.env.step([0, -0.2 * self.twist_range, 0])
            # secondly move forward until there is an obstacle or reach the goal
            sounder = self.obs_img_ray[1]
            # echo sounder readings: 1. middle 2. first right 3. first left 4. second right ......
            self.multibeam = np.array([sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                        sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19], sounder[23]]) * 4 # from left to right

            while distance > 0.8 and np.min(self.multibeam) > 1.5:
                obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                distance = obs_goal_depthfromwater[0]
                self.obs_img_ray, _, done, _ = self.env.step([0, 0, 1])
                sounder = self.obs_img_ray[1]
                self.multibeam = np.array([sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19], sounder[23]]) * 4
            if distance < 1:
                break

            # follow the wall until the robot reaches the intersection line
            # turn its direction
            index_init = np.argmin(self.multibeam)
            index = index_init
            if index_init < 7:
                while index != 0:
                    # obstacle is on robot's left, turn right
                    self.obs_img_ray, _, done, _ = self.env.step([0, -0.5 * self.twist_range, 0])
                    sounder = self.obs_img_ray[1]
                    self.multibeam = np.array([sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                                 sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19], sounder[23]])*4
                    index = np.argmin(self.multibeam)
            else:
                while index != len(self.multibeam) - 1: # turn left
                    self.obs_img_ray, _, done, _ = self.env.step([0, 0.5 * self.twist_range, 0])
                    sounder = self.obs_img_ray[1]
                    self.multibeam = np.array([sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                                 sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19], sounder[23]])*4
                    index = np.argmin(self.multibeam)

            obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
            cur_pos = np.array([obs_goal_depthfromwater[4], obs_goal_depthfromwater[5]])
            goal_pos = np.array([self.start_goal_pos[6], self.start_goal_pos[8]])

            print("0\n\n\n")
            if index_init < 7:
                # follow the wall by turning left
                if (goal_pos - cur_pos)[0] > 0:
                    slope = (goal_pos - cur_pos)[1] / (goal_pos - cur_pos)[0]
                    intercept = goal_pos[1] - slope * goal_pos[0]
                    y_comp = slope * cur_pos[0] + intercept
                    while .05 + y_comp > cur_pos[1]:
                        if np.min(self.multibeam) == 4:
                            self.obs_img_ray, _, done, _ = self.env.step([0, 0.5 * self.twist_range, 0.5])
                            obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                            cur_pos = np.array([obs_goal_depthfromwater[4], obs_goal_depthfromwater[5]])
                            sounder = self.obs_img_ray[1]
                            self.multibeam = np.array(
                                [sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                                 sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19],
                                 sounder[23]]) * 4
                            y_comp = slope * cur_pos[0] + intercept
                        elif np.argmin(self.multibeam) != 0:
                            self.obs_img_ray, _, done, _ = self.env.step([0, -0.5 * self.twist_range, 0.5])
                            obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                            cur_pos = np.array([obs_goal_depthfromwater[4], obs_goal_depthfromwater[5]])
                            sounder = self.obs_img_ray[1]
                            self.multibeam = np.array(
                                [sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                                 sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19],
                                 sounder[23]]) * 4
                            y_comp = slope * cur_pos[0] + intercept
                        else:
                            self.obs_img_ray, _, done, _ = self.env.step([0, 0.5 * self.twist_range, 0.5])
                            obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                            cur_pos = np.array([obs_goal_depthfromwater[4], obs_goal_depthfromwater[5]])
                            sounder = self.obs_img_ray[1]
                            self.multibeam = np.array(
                                [sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                                 sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19],
                                 sounder[23]]) * 4
                            y_comp = slope * cur_pos[0] + intercept
                elif (goal_pos - cur_pos)[0] < 0:
                    slope = (goal_pos - cur_pos)[1] / (goal_pos - cur_pos)[0]
                    intercept = goal_pos[1] - slope * goal_pos[0]
                    y_comp = slope * cur_pos[0] + intercept
                    while y_comp < cur_pos[1] + 0.05:
                        if np.min(self.multibeam) == 4:
                            self.obs_img_ray, _, done, _ = self.env.step([0, 0.5 * self.twist_range, 0.5])
                            obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                            cur_pos = np.array([obs_goal_depthfromwater[4], obs_goal_depthfromwater[5]])
                            sounder = self.obs_img_ray[1]
                            self.multibeam = np.array(
                                [sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                                 sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19],
                                 sounder[23]]) * 4
                            y_comp = slope * cur_pos[0] + intercept
                        elif np.argmin(self.multibeam) != 0:
                            self.obs_img_ray, _, done, _ = self.env.step([0, -0.5 * self.twist_range, 0.5])
                            obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                            cur_pos = np.array([obs_goal_depthfromwater[4], obs_goal_depthfromwater[5]])
                            sounder = self.obs_img_ray[1]
                            self.multibeam = np.array(
                                [sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                                 sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19],
                                 sounder[23]]) * 4
                            y_comp = slope * cur_pos[0] + intercept
                        else:
                            self.obs_img_ray, _, done, _ = self.env.step([0, 0.5 * self.twist_range, 0.5])
                            obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                            cur_pos = np.array([obs_goal_depthfromwater[4], obs_goal_depthfromwater[5]])
                            sounder = self.obs_img_ray[1]
                            self.multibeam = np.array(
                                [sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                                 sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19],
                                 sounder[23]]) * 4
                            y_comp = slope * cur_pos[0] + intercept

            else:
                print("1\n\n\n")
                # follow the wall by turning right
                if (goal_pos - cur_pos)[0] > 0:
                    slope = (goal_pos - cur_pos)[1] / (goal_pos - cur_pos)[0]
                    intercept = goal_pos[1] - slope * goal_pos[0]
                    y_comp = slope * cur_pos[0] + intercept
                    while  y_comp < .05 + cur_pos[1]:
                        if np.min(self.multibeam) == 4:
                            self.obs_img_ray, _, done, _ = self.env.step([0, -0.5 * self.twist_range, 0.5])
                            obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                            cur_pos = np.array([obs_goal_depthfromwater[4], obs_goal_depthfromwater[5]])
                            sounder = self.obs_img_ray[1]
                            self.multibeam = np.array(
                                [sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                                 sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19],
                                 sounder[23]]) * 4
                            y_comp = slope * cur_pos[0] + intercept
                        elif np.argmin(self.multibeam) != 12:
                            self.obs_img_ray, _, done, _ = self.env.step([0, 0.5 * self.twist_range, 0.5])
                            obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                            cur_pos = np.array([obs_goal_depthfromwater[4], obs_goal_depthfromwater[5]])
                            sounder = self.obs_img_ray[1]
                            self.multibeam = np.array(
                                [sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                                 sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19],
                                 sounder[23]]) * 4
                            y_comp = slope * cur_pos[0] + intercept
                        else:
                            self.obs_img_ray, _, done, _ = self.env.step([0, -0.5 * self.twist_range, 0.5])
                            obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                            cur_pos = np.array([obs_goal_depthfromwater[4], obs_goal_depthfromwater[5]])
                            sounder = self.obs_img_ray[1]
                            self.multibeam = np.array(
                                [sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                                 sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19],
                                 sounder[23]]) * 4
                            y_comp = slope * cur_pos[0] + intercept
                elif (goal_pos - cur_pos)[0] < 0:
                    slope = (goal_pos[1] - cur_pos[1]) / (goal_pos[0] - cur_pos[0])
                    intercept = goal_pos[1] - slope * goal_pos[0]
                    y_comp = slope * cur_pos[0] + intercept
                    print(slope, "\n\n\n")
                    while y_comp > cur_pos[1] - 0.05:
                        print(goal_pos, cur_pos, slope, intercept, y_comp, slope * cur_pos[0] + intercept, obs_goal_depthfromwater[6])
                        if np.min(self.multibeam) == 4:
                            self.obs_img_ray, _, done, _ = self.env.step([0, -0.5 * self.twist_range, 0.5])
                            obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                            cur_pos = np.array([obs_goal_depthfromwater[4], obs_goal_depthfromwater[5]])
                            sounder = self.obs_img_ray[1]
                            self.multibeam = np.array(
                                [sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                                 sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19],
                                 sounder[23]]) * 4
                            y_comp = slope * cur_pos[0] + intercept
                        elif np.argmin(self.multibeam) != 12:
                            self.obs_img_ray, _, done, _ = self.env.step([0, 0.5 * self.twist_range, 0.5])
                            obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                            cur_pos = np.array([obs_goal_depthfromwater[4], obs_goal_depthfromwater[5]])
                            sounder = self.obs_img_ray[1]
                            self.multibeam = np.array(
                                [sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                                 sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19],
                                 sounder[23]]) * 4
                            y_comp = slope * cur_pos[0] + intercept
                        else:
                            self.obs_img_ray, _, done, _ = self.env.step([0, -0.5 * self.twist_range, 0.5])
                            obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
                            cur_pos = np.array([obs_goal_depthfromwater[4], obs_goal_depthfromwater[5]])
                            sounder = self.obs_img_ray[1]
                            self.multibeam = np.array(
                                [sounder[25], sounder[21], sounder[17], sounder[13], sounder[9], sounder[5], \
                                 sounder[1], sounder[3], sounder[7], sounder[11], sounder[15], sounder[19],
                                 sounder[23]]) * 4
                            y_comp = slope * cur_pos[0] + intercept
            print("how???\n\n\n")
        print("how???\n\n\n")
        obs_preddepth = self.dpt.run(self.obs_img_ray[0] ** 0.45)
        # obs_ray = np.array([np.min([obs_img_ray[1][1], obs_img_ray[1][3], obs_img_ray[1][5],
        #                             obs_img_ray[1][33], obs_img_ray[1][35]]) * 8 * 0.5])

        obs_goal_depthfromwater = self.pos_info.goal_depthfromwater_info()

        """
            compute reward
            obs_goal_depthfromwater[0]: horizontal distance
            obs_goal_depthfromwater[1]: vertical distance
            obs_goal_depthfromwater[2]: angle from robot's orientation to the goal (degree)
            obs_goal_depthfromwater[3]: robot's current y position
            obs_goal_depthfromwater[4]: robot's current x position            
            obs_goal_depthfromwater[5]: robot's current z position            
        """
        # 1. give a negative reward when robot is too close to nearby obstacles, seafloor or the water surface
        obstacle_distance = np.min([self.obs_img_ray[1][1], self.obs_img_ray[1][3], self.obs_img_ray[1][5],
                             self.obs_img_ray[1][7], self.obs_img_ray[1][9], self.obs_img_ray[1][11],
                             self.obs_img_ray[1][13], self.obs_img_ray[1][15], self.obs_img_ray[1][17]]) * 8 * 0.5
        obstacle_distance_vertical = 3
        if obstacle_distance < 0.5 or np.abs(obs_goal_depthfromwater[3]) < 0.24 or obstacle_distance_vertical < 0.12:
            reward_obstacle = -10
            done = True
            print("Too close to the obstacle, seafloor or water surface!",
                  "\nhorizontal distance to nearest obstacle:", obstacle_distance,
                  "\ndistance to water surface", np.abs(obs_goal_depthfromwater[3]),
                  "\nvertical distance to nearest obstacle:", obstacle_distance_vertical)
        else:
            reward_obstacle = 0

        # 2. give a positive reward if the robot reaches the goal
        if self.training:
            if obs_goal_depthfromwater[0] < 0.6:
                reward_goal_reached = 10 - 8 * np.abs(obs_goal_depthfromwater[1]) - np.abs(np.deg2rad(obs_goal_depthfromwater[2]))
                done = True
                print("Reached the goal area!")
            else:
                reward_goal_reached = 0
        else:
            if self.waypoint != 4:
                if obs_goal_depthfromwater[0] < 0.8:
                    reward_goal_reached = 10 - 8 * np.abs(obs_goal_depthfromwater[1]) - np.abs(np.deg2rad(obs_goal_depthfromwater[2]))
                    done = True
                    self.waypoint += 1
                    print("Reached the goal area!")
                    if self.waypoint < 5:
                        my_open = open(os.path.join(assets_dir(), 'learned_models/test_pos.txt'), "a")
                        data = ["\n"]
                        for element in data:
                            my_open.write(element)
                        my_open.close()
                        done = False
                        self.pos_info.assign_testpos_visibility(
                            [obs_goal_depthfromwater[4], obs_goal_depthfromwater[3], obs_goal_depthfromwater[5]
                                , 0, obs_goal_depthfromwater[6], 0] +
                            self.start_goal_pos[9 + 3 * (self.waypoint - 1): 9 + 3 * self.waypoint] + [20])

                else:
                    reward_goal_reached = 0

            else:
                if obs_goal_depthfromwater[0] < 10.:
                    reward_goal_reached = 10 - 8 * np.abs(obs_goal_depthfromwater[1]) - np.abs(np.deg2rad(obs_goal_depthfromwater[2]))
                    done = True
                    self.waypoint += 1
                    print("Reached the goal area!")
                    if self.waypoint < 5:
                        my_open = open(os.path.join(assets_dir(), 'learned_models/test_pos.txt'), "a")
                        data = ["\n"]
                        for element in data:
                            my_open.write(element)
                        my_open.close()
                        done = False
                        self.pos_info.assign_testpos_visibility([obs_goal_depthfromwater[4], obs_goal_depthfromwater[3], obs_goal_depthfromwater[5]
                                                                 , 0, obs_goal_depthfromwater[6], 0] +
                                                                self.start_goal_pos[9 + 3 * (self.waypoint -1) : 9 + 3 * self.waypoint] + [20])
                else:
                    reward_goal_reached = 0

        return self.obs_preddepths, self.obs_goals, self.obs_rays, self.obs_actions, 0, done, 0
