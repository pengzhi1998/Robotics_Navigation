import os
from utils import *

cwd = os.path.join(os.getcwd(), "./ppo_gym_test.py")
env = ['test3', 'test5', 'test6']
vis = ['3000', '2000', '1000']
model = ['C', 'B', 'A']
for i in env:
    for j in vis:
        for k in model:
            my_open = open(os.path.join(assets_dir(), 'learned_models/test_pos.txt'), "a")
            data = [i + "_" + j + "_" + k, "\n"]
            for element in data:
                my_open.write(element)
            my_open.close()
            os.system('{} {} --env-name=navigation --hist-length=5 --depth-prediction-model=midas --scene-name={} '
                      '--visibility={} --model-path={} --eval-batch-size=20000 '
                      '--adaptation=0 --randomization=0'.format('python3', cwd, i, j, k))
            my_open = open(os.path.join(assets_dir(), 'learned_models/test_pos.txt'), "a")
            data = ["\n"]
            for element in data:
                my_open.write(element)
            my_open.close()