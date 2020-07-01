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

from A2C import Agent
from Environments import Car_Env
from Resources.Utils import score_plot

def main():

    ################ USER INPUT ################

    test_mode = False
    load_checkpoint = False

    render_on = False

    env_name = "CartPole-v0"

    ############################################
    alg_name = "A2C"

    if env_name=="car":
        env = Car_Env.MainEnv( rand_start = True )
    else:
        env = gym.make(env_name)

    ## We get the action and input shape from the environments themselves
    inp_space = list( env.reset().shape )
    act_space = env.action_space.n

    agent = Agent(
                    name    = env_name + "_" + alg_name,
                    net_dir = home_env + "Saved_Models",
                    \
                    gamma = 0.99, lr = 1e-3,
                    \
                    input_dims = inp_space, n_actions = act_space,
                    depth = 3, width = 64, activ = nn.PReLU(),
                    \
                    env_name = env_name,
                    n_workers = 32, n_frames = 8,
                    vf_coef = 0.1, ent_coef = 0.0001,
                    )

    if load_checkpoint:
        agent.load_model()

    for t in count():

        if render_on:
            agent.workers[0].env.render()

        loss = agent.train()

        if not test_mode:
            if t>=1000 and t%1000==0:
                agent.save_model()

        print( "Iteration = {}, Loss = {:4.3f}        \r".format( t, loss ), end="" )
        sys.stdout.flush()


if __name__ == '__main__':
    main()
