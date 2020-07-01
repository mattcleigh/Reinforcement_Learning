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

from PPO import Agent
from Environments import Car_Env
from Resources.Utils import score_plot

def main():

    ################ USER INPUT ################

    test_mode = False
    load_checkpoint = False

    render_on = False

    env_name = "LunarLander-v2"

    ############################################
    alg_name = "PPO"

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
                    gamma = 0.99, lr = 1e-5,
                    \
                    input_dims = inp_space, n_actions = act_space,
                    depth = 3, width = 64, activ = nn.PReLU(),
                    \
                    eps_clip = 0.2, pol_sync = 4,
                    \
                    env_name = env_name,
                    n_workers = 4, n_frames = 32,
                    vf_coef = 0.1, ent_coef = 0.0001,
                    )

    if load_checkpoint:
        agent.load_model()

    for t in count():

        if render_on:
            agent.workers[0].env.render()

        loss = agent.train()

        if not test_mode:
            if t>=500 and t%500==0:
                agent.save_models()

        print( "Iteration = {}, Loss = {:4.3f}        \r".format( t, loss ), end="" )
        sys.stdout.flush()


if __name__ == '__main__':
    main()
