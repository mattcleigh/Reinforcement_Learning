import sys
home_env = '../../../../Reinforcement_Learning/'
sys.path.append(home_env)

import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

import torch as T
import torch.nn as nn

from Environments import Car_Env
from Resources import Utils as myUT

from PPO import Agent

def main():

    ################ USER INPUT ################

    test_mode = False
    load_prev = False
    render_on = True

    save_every = 10

    env_name = "car"
    alg_name = "PPO"

    ############################################

    ## Loading the environment
    if env_name=="car":
        env = Car_Env.MainEnv( rand_start = True )
    else:
        env = gym.make(env_name)

    ## We get the action and input shape from the environments themselves
    inp_space = list( env.reset().shape )
    act_space = env.action_space.n

    agent = Agent(
                    name    = alg_name + "_" + env_name,
                    net_dir = home_env + "Saved_Models/" + alg_name,
                    \
                    gamma = 0.99, lr = 1e-3, grad_clip = 10,
                    \
                    input_dims = inp_space, n_actions = act_space,
                    depth = 3, width = 128, activ = nn.ReLU(),
                    \
                    eps_clip = 0.2, pol_sync = 50,
                    \
                    env_name = env_name,
                    n_workers = 4, n_frames = 512,
                    vf_coef = 0.1, ent_coef = 0.01,
                    )

    if load_prev:
        agent.load_models()

    myUT.train_ac_model(agent, env, render_on, test_mode, save_every)

if __name__ == '__main__':
    main()
