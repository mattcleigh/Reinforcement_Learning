import sys
home_env = '../../../../Reinforcement_Learning/'
sys.path.append(home_env)

import gym
import time
import pybullet_envs
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

import torch as T
import torch.nn as nn

from Resources import Utils as myUT

from TD3 import Agent

def main():

    ################ USER INPUT ################

    test_mode = False
    load_prev = False
    render_on = True
    save_every = 10000

    env_name = "Walker2DBulletEnv-v0"
    alg_name = "TD3"

    ############################################

    ## Loading the environment
    env = gym.make(env_name)
    env.render()

    ## We get the action and input shape from the environments themselves
    inp_space = list(env.observation_space.shape)
    act_space = env.action_space.shape[0]

    agent = Agent(
                    name    = alg_name + "_" + env_name,
                    net_dir = home_env + "Saved_Models/" + alg_name,
                    \
                    gamma = 0.99,
                    input_dims = inp_space, n_actions = act_space,
                    active = nn.ReLU(), grad_clip = 0, QL2 = 0,
                    noisy = False,
                    \
                    C_lr = 1e-3, C_depth = 2, C_width = 400,
                    A_lr = 1e-3, A_depth = 2, A_width = 400,
                    \
                    eps = 1e-1, eps_min = 1e-3, eps_dec = 1e-6,
                    \
                    delay = 2, smooth_noise = 0.2, noise_clip = 0.5,
                    \
                    mem_size = 1000000,   batch_size = 100,
                    target_sync = 5e-3,  freeze_up = 5000,
                    \
                    PER_on    = True, n_step  = 3,
                    PEReps    = 0.01, PERa     = 0.6,
                    PERbeta   = 0.4,  PERb_inc = 1e-6,
                    PERmax    = 1,
                    )

    if load_prev:
        agent.load_models()

    myUT.train_dqn_model( agent, env, render_on, test_mode, save_every )


if __name__ == '__main__':
    main()
