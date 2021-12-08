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



