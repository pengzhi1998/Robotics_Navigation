import argparse
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils import *
from core.agent import Agent
from core.unity_underwater_env import Underwater_navigation

parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-5, metavar='G',
                    help='learning rate (default: 3e-5)')
parser.add_argument('--randomization', type=int, default=1, metavar='G')
parser.add_argument('--adaptation', type=int, default=1, metavar='G')
parser.add_argument('--depth-prediction-model', default="dpt", metavar='G')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--hist-length', type=int, default=4, metavar='N',
                    help="the number of consecutive history infos (default: 4)")
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--start-goal-pos', type=list, default=[8., -2., -0.376, 0., 270., 0., -3.86, -2.3, -4.15],
                    metavar='N', help="during testing, what are the initial positions, orientations for the"
                                      "robot and the initial positions for the goal point")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
args = parser.parse_args()

"""device"""
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""
env = []
# start_goal_pos = [15., -2.5, -15, 0., 270., 0., -20, -1.5, 20] # test0
start_goal_pos = [15., -2.5, -15, 0., 270., 0., 5, -1.5,-5] # test0
# start_goal_pos = [14, -3.42, 0., 0., 0., 0., -6.67, -1.97, -3.45] # test3
# start_goal_pos = [8, -3.42, 2.76, 0., 270., 0., -7.67, -1.97, 1.45] # test5
# start_goal_pos = [10, -1.5, 0, 0., 270., 0., -10, -3.5, 0] # test6

# for the paper's image
# start_goal_pos = [9, -3.42, -1., 0., 270., 0., -6.67, -1.97, -3.45] # test3
# start_goal_pos = [4.367, -1.81, -0.63, 0., 270., 0., -7.67, -1.97, 1.45] # test5
# start_goal_pos = [5.2, -4.37, 4.77, 0., 250., 0., -10, -3.5, 0] # test6


for i in range(args.num_threads):
    env.append(Underwater_navigation(args.depth_prediction_model, args.adaptation, args.randomization, i, args.hist_length,
                                     start_goal_pos, False))
img_depth_dim = env[0].observation_space_img_depth
goal_dim = env[0].observation_space_goal
ray_dim = env[0].observation_space_ray

"""define actor and critic"""
if args.randomization == True:
    policy_net, value_net, running_state = pickle.load(
        open(os.path.join(assets_dir(), 'learned_models/{}_ppo_rand_best.p'.format(args.env_name,
                                                                      args.hist_length)), "rb"))
else:
    policy_net, value_net, running_state = pickle.load(
        open(os.path.join(assets_dir(), 'learned_models/{}_ppo_norand_10_250iters.p'.format(args.env_name,
        # open(os.path.join(assets_dir(), 'learned_models/{}_ppo_norand_2000_250iters.p'.format(args.env_name,
        # open(os.path.join(assets_dir(), 'learned_models/{}_ppo_rand_noechosounder_250iters.p'.format(args.env_name,
        # open(os.path.join(assets_dir(), 'learned_models/{}_ppo_rand_250iters.p'.format(args.env_name,
                                                                              args.hist_length)), "rb"))

policy_net.to(device)

"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, num_threads=args.num_threads, training=False)

while True:
    if args.eval_batch_size > 0:
        _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)