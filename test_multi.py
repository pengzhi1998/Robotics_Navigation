import time
import uuid
from utils import *

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import multiprocessing
import torch
from models.mlp_policy import Policy
from core.unity_underwater_env import Underwater_navigation

from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
)
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

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

def create_env(rank):
    # unity_env = UnityEnvironment(os.path.abspath("./") + "/basic_test", worker_id=rank, base_port=5000+rank)
    # unity_env = UnityEnvironment("/home/pengzhi1998/navigation/mega_navigation/Robotics_Navigation/underwater_env/water"
    #                              , worker_id=rank, base_port=5000+rank)
    # env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
    pos_info = PosChannel()
    config_channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment("/home/pengzhi1998/navigation/mega_navigation/Robotics_Navigation/underwater_env/water",
                                 side_channels=[config_channel, pos_info], worker_id=rank, base_port=5000 + rank)
    config_channel.set_configuration_parameters(time_scale=10, capture_frame_rate=100)
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

    return env

def collect_samples(pid, queue, env, policy, custom_reward, mean_action, render, running_state, thread_batch_size):
    THRESHOLD = torch.tensor(np.finfo("float").eps).to(device)
    print(pid, "collect_sample\n\n\n")
    done = False
    env.reset()
    while done is False:
        print(pid, "collect_sample\n\n\n")
        obsimg, obsgoal, obsray, reward, done, _ = env.step([0,1])

    if queue != None:
        queue.put([1])
    return pid

class Agent:

    def __init__(self, env, policy, device, custom_reward=None, running_state=None, num_threads=2):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.running_state = running_state
        self.num_threads = num_threads

    def collect_samples(self, min_batch_size, mean_action=False, render=False):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        print("c")
        for i in range(self.num_threads-1):
            env = self.env[i+1]
            worker_args = (i+1, queue, env, self.policy, self.custom_reward, mean_action,
                           False, self.running_state, thread_batch_size)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            print("d")
            worker.start()
            print("e")

        memory, log = collect_samples(0, None, self.env[0], self.policy, self.custom_reward, mean_action,
                                      render, self.running_state, thread_batch_size)
                                      # render, None, thread_batch_size)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    env = []
    for i in range(2):
        # env.append(create_env(i))
        env.append(Underwater_navigation(i))

    device = torch.device("cuda")
    policy_net = Policy(env[0].action_space.shape[0], log_std=0)
    policy_net.to(device)
    img_depth_dim = env[0].observation_space_img_depth
    goal_dim = env[0].observation_space_goal
    ray_dim = env[0].observation_space_ray
    running_state = ZFilter(img_depth_dim, goal_dim, ray_dim, clip=30)

    agent = Agent(env, policy_net, device, custom_reward=None, running_state=None, num_threads=2)

    agent.collect_samples(2048, mean_action=False, render=False)

    # workers = []
    # queue = multiprocessing.Queue()
    # for i in range(1):
    #     worker_args = (i + 1, env[i + 1], queue, device, policy_net, running_state)
    #     workers.append(multiprocessing.Process(target=collect_sample, args=worker_args))
    # for worker in workers:
    #     print("d")
    #     worker.start()
    #     print("e")
    #
    # collect_sample(0, env[0], None, device, policy_net, running_state)
    print("end!")
