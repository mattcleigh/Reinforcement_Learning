import sys
home_env = '../../../Reinforcement_Learning/'
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

from A2C import Agent

def main():

    ################ USER INPUT ################

    test_mode = False
    load_prev = False
    render_on = False

    save_every = 500

    env_name = "CartPole-v0"
    alg_name = "A2C"

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
                    gamma = 0.99, lr = 1e-4, grad_clip = 0,
                    \
                    input_dims = inp_space, n_actions = act_space,
                    depth = 3, width = 64, activ = nn.PReLU(),
                    \
                    env_name = env_name,
                    n_workers = 16, n_frames = 32,
                    vf_coef = 0.1, ent_coef = 0.01,
                    )

    if load_prev:
        agent.load_models()

    myUT.train_ac_model(agent, env, render_on, test_mode, save_every)


if __name__ == '__main__':
    main()
