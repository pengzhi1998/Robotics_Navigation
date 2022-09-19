## Underwater Navigation
Source code for paper *Monocular Camera and Single-Beam Sonar-Based Underwater Collision-Free 
Navigation with Domain Randomization*.
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