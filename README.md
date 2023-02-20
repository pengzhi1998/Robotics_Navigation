## Underwater Navigation
Source code for paper [*Monocular Camera and Single-Beam Sonar-Based Underwater Collision-Free 
Navigation with Domain Randomization*](https://arxiv.org/abs/2212.04373). You could also check [this](https://github.com/dartmouthrobotics/deeprl-uw-robot-navigation) as another github repo.
The PPO implementation and the underwater env are referred to [PyTorch-RL](https://github.com/Khrylx/PyTorch-RL.git), and 
[Optically-Realistic-Water](https://github.com/muckSponge/Optically-Realistic-Water) respectively.
### Installation
```
virtualenv --no-site-packages UW_nav --python=python3.6
source ~/UW_nav/bin/activate
pip install -r requirements.txt
cd ./
git clone --branch release_18 https://github.com/Unity-Technologies/ml-agents.git
cd ./ml-agents
pip install -e ./ml-agents-envs
pip install gym-unity==0.27.0
```
To build the Unity environment, clone [this repository](https://github.com/hdacnw/Underwater-RL/tree/b0710e3b79a579b66a157429658b3418d5b2b739)
(indicated as a submodule named *Unity* in our repository). Build the **SampleScene** and
choose `./Robotics_Navigation/underwater_env/` as the path. Besides, download [dpt_large-midas-2f21e586.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt)
to the path `./Robotics_Navigation/DPT/weights`.

### Training
<!--
## Dependencies
Ubuntu 18.04, ROS Melodic, python 3.6.9, cuda 10.2

## How to run the code
(1) Clone the repository

(2) Set up the environments
* Make up your ROS catkin space. And
we were using a Turtlebot to train the policy.
For next step, we'll use UUVSimulator to train 
it instead.
* Copy the designed world `empty.world`
 and launch file `turtlebot3_empty_world.launch` from 
 `assest/ROS` into the worlds directory
 and launch file directory respectively.
* Install the dependencies for training.

(3) Run the code
* Launch Gazebo in one terminal:
 
 $ roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch

* In another terminal, run the code:
 
 $ export OMP_NUM_THREADS=1
 $ python ppo_gym.py --save-model-interval 5 --env-name navigation --eval-batch-size 0 --min-batch-size 2048

(4) In the directory of assets, you would find the trained model and the log file. 
 
 (4) After training, you could use `DDDQN_test.py`
and `DDDQN_uwsim.py` to test the performance in Gazebo
worlds and UWSim worlds respectively. For real-world
tests, refer to [this](https://github.com/pengzhi1998/underwater_navigation_test)
 repository. -->
 ```
 python3 ppo_gym.py --save-model-interval 5 --env-name navigation --eval-batch-size 0 --min-batch-size 2048 --num-threads 1 --hist-length 5
```
### Testing

[//]: # (Remember to modify the threshold value to 0.5m and 0.25m:)
```
python3 ppo_gym_test.py --env-name navigation --eval-batch-size 2000 --hist-length 5
```
