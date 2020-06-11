from collections import deque
from typing import Deque, Dict, List, Tuple

import sys
sys.path.append('/home/matthew/Documents/Reinforcement_Learning/')

import time
import math
import matplotlib.pyplot as plt
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

from Environments import Car_Env

def main():

    ## Create the driving environment
    env = Car_Env.MainEnv()

    for ep in count():

        state = env.reset()

        for it in count():
            env.render()

            next_state, reward, failed, _ = env.step(1)

            ## We exit the loop when the fail state is reached
            if failed:
                break


if __name__ == '__main__':
    main()












