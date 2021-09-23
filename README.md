# Training
## Dependencies
python 2.7.17 cuda 10.1, 10.0 torch 1.4.0

## How to run the code
1. clone the repository
2. Set up the environments (virtualenv or docker)

3. Use the designed world and 
``` $ python DDDQN.py ```

## What you'll see
In the terminal, the relative goal info will be printed out. The first element for the array is the relative distance, 
second one is the relative angle (in radian system, positive value means the goal is on robot's left, negative value means the goal is on robot's right). 
The control command will be printed out as well and if twist value less than 1500, it means turning left, while value bigger than 1500 means turning right.
