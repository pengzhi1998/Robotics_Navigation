import argparse
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils import *
from models.mlp_policy import Policy
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
from unity_underwater_env import Underwater_navigation

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
for i in range(args.num_threads):
    env.append(Underwater_navigation(i, args.hist_length))
img_depth_dim = env[0].observation_space_img_depth
goal_dim = env[0].observation_space_goal
ray_dim = env[0].observation_space_ray

"""define actor and critic"""
policy_net, value_net, running_state = pickle.load(
open(os.path.join(assets_dir(), 'learned_models/{}_ppo_6.p'.format(args.env_name)), "rb")
)
policy_net.to(device)

"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, num_threads=args.num_threads)

while True:
    if args.eval_batch_size > 0:
        _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)