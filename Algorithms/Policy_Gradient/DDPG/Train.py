import sys
home_env = '../../../'
sys.path.append(home_env)

import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

import torch as T
import torch.nn as nn

from Resources import Utils as myUT

from DDPG import Agent

def main():

    ################ USER INPUT ################

    test_mode = False
    load_prev = False
    render_on = True
    save_every = 10000

    env_name = "LunarLanderContinuous-v2"
    alg_name = "DDPG"

    ############################################

    ## Loading the environment
    env = gym.make(env_name)

    ## We get the action and input shape from the environments themselves
    inp_space = list(env.observation_space.shape)
    act_space = env.action_space.shape[0]

    agent = Agent(
                    name    = alg_name + "_" + env_name,
                    net_dir = home_env + "Saved_Models/" + alg_name,
                    \
                    gamma = 0.99,
                    input_dims = inp_space, n_actions = act_space,
                    active = nn.ReLU(), grad_clip = 10, QL2 = 1e-2,
                    noisy = False,
                    \
                    C_lr = 1e-3, C_depth = 2, C_width = 400,
                    A_lr = 1e-3, A_depth = 2, A_width = 400,
                    \
                    eps = 2e-1, eps_min = 1e-3, eps_dec = 2e-6,
                    \
                    mem_size = 100000,   batch_size = 64,
                    target_sync = 5e-3,  freeze_up = 100,
                    \
                    PER_on    = False, n_step  = 1,
                    PEReps    = 0.01, PERa     = 0.6,
                    PERbeta   = 0.4,  PERb_inc = 1e-6,
                    PERmax    = 1,
                    )

    if load_prev:
        agent.load_models()

    myUT.train_dqn_model( agent, env, render_on, test_mode, save_every )


if __name__ == '__main__':
    main()
