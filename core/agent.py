import multiprocessing
from utils.replay_memory import Memory
from utils.torchpy import *
from utils.tools import *
import math
import time
import os
import sys
import signal
os.environ["OMP_NUM_THREADS"] = "1"

def signal_handler(sig, frame):
    sys.exit(0)

def collect_samples(pid, queue, env, policy, custom_reward,
                    mean_action, render, running_state, min_batch_size, training=True):
    if pid > 0:
        torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
        if hasattr(env, 'np_random'):
            env.np_random.seed(env.np_random.randint(5000) * pid)
        if hasattr(env, 'env') and hasattr(env.env, 'np_random'):
            env.env.np_random.seed(env.env.np_random.randint(5000) * pid)
    log = dict()
    memory = Memory()
    num_steps = 0
    num_episodes_success = 0
    num_steps_episodes = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0
    reward_done = 0

    print(time.time())

    while num_steps < min_batch_size:
        img_depth, goal, ray, hist_action = env.reset()
        if running_state is not None:
            # print "before", img_depth.shape, goal.shape, img_depth.dtype
            # print "first_depth_before:", np.max(img_depth), "first_goal_before:", np.max(goal)
            # print(np.shape(img_depth), np.shape(goal), np.shape(ray), np.shape(hist_action), "\n\n\n")
            _, goal, ray = running_state(img_depth, goal, ray)
            img_depth = np.float64((img_depth - 0.5) / 0.5) # the predicted depth ranges from 0 - 1
            hist_action = np.float64(hist_action)
            # print "first_depth_after:", np.max(img_depth), "first_goal_after:", np.max(goal)
            # print "after", img_depth.shape, goal.shape, img_depth.dtype
        else:
            img_depth, goal, ray, hist_action = \
                img_depth.astype(np.float64), goal.astype(np.float64), ray.astype(np.float64), \
                hist_action.astype(np.float64)
        reward_episode = 0

        for t in range(10000):
            # print t
            signal.signal(signal.SIGINT, signal_handler)
            img_depth_var = tensor(img_depth).unsqueeze(0)
            goal_var = tensor(goal).unsqueeze(0)
            ray_var = tensor(ray).unsqueeze(0)
            hist_action_var = tensor(hist_action).unsqueeze(0)
            with torch.no_grad():
                if mean_action:
                    action = policy(img_depth_var, goal_var, ray_var, hist_action_var)[0][0].numpy()
                else:
                    action = policy.select_action(img_depth_var, goal_var, ray_var, hist_action_var)[0].numpy()
            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            next_img_depth, next_goal, next_ray, next_hist_action, reward, done, _ = env.step(action)
            reward_episode += reward
            if running_state is not None:
                # print "before", next_img_depth.shape, next_goal.shape
                # print "depth_before:", np.max(next_img_depth), np.min(next_img_depth), "goal_before:", np.max(next_goal), np.min(goal)
                _, next_goal, next_ray = running_state(next_img_depth, next_goal, next_ray)
                next_img_depth = np.float64((next_img_depth - 0.5) / 0.5)
                next_hist_action = np.float64(next_hist_action)
                # print next_img_depth
                # print "depth_after:", np.max(next_img_depth), np.min(next_img_depth), "goal_after:", np.max(next_goal), np.min(goal), "\n\n\n"
                # print "after", next_img_depth.shape, next_goal.shape
            else:
                next_img_depth, next_goal, next_ray, next_hist_action = \
                    next_img_depth.astype(np.float64), next_goal.astype(np.float64),\
                    next_ray.astype(np.float64), next_hist_action.astype(np.float64)

            if custom_reward is not None:
                reward = custom_reward(img_depth, goal, ray, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1

            memory.push(img_depth, goal, ray, hist_action, action, mask,
                        next_img_depth, next_goal, next_hist_action, reward)

            if render:
                env.render()
            if done:
                reward_done += reward
                if reward > 0 and t < 499:
                    num_episodes_success += 1
                    num_steps_episodes += t
                break

            img_depth = next_img_depth
            goal = next_goal
            ray = next_ray
            hist_action = next_hist_action

        # log stats
        num_steps += (t + 1)
        # print "num_steps:", num_steps
        num_episodes += 1
        total_reward += reward_episode
        print("reward for one episode", reward_episode, "\n")
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

        if training == False:
            my_open = open(os.path.join(assets_dir(), 'learned_models/test_pos.txt'), "a")
            data = [str(reward_episode), "\n\n"]
            for element in data:
                my_open.write(element)
            my_open.close()
            if num_episodes >= 5:
                exit()

    print(time.time())

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    log['num_episodes'] = num_episodes
    log['ratio_success'] = float(num_episodes_success) / float(num_episodes)
    log['avg_last_reward'] = reward_done / num_episodes
    if num_episodes_success != 0:
        log['avg_steps_success'] = float(num_steps_episodes) / float(num_episodes_success)
    else:
        log['avg_steps_success'] = 0

    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env, policy, device, custom_reward=None, running_state=None, num_threads=1, training=True):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.running_state = running_state
        self.num_threads = num_threads
        self.training = training

    def collect_samples(self, min_batch_size, mean_action=False, render=False):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            env = self.env[i+1]
            worker_args = (i+1, queue, env, self.policy, self.custom_reward, mean_action,
                           False, self.running_state, thread_batch_size, self.training)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env[0], self.policy, self.custom_reward, mean_action,
                                      render, self.running_state, thread_batch_size, training=self.training)
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
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log
