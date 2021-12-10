import torch
import numpy as np
import time
device = torch.device('cuda')
while True:
    time0 = time.time()
    a = torch.rand(1920, 1080).to(device)
    a = a.cpu()
    time1 = time.time()
    b = torch.tensor([3,2]).to(device)
    b = b.cpu()
    time2 = time.time()
    print(time2 - time1, time1 - time0)