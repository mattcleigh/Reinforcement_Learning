import sys
home_env = '../../../Reinforcement_Learning/'
sys.path.append(home_env)

import gym
import torch.nn as nn

from Environments import Car_Env
from Resources import Utils as myUT

from QRDQN import Agent

def main():

    ################ USER INPUT ################

    test_mode = False
    load_prev = False
    render_on = True
    save_every = 10000

    draw_return = True
    draw_interv = 10

    env_name = "CartPole-v0"
    alg_name = "QRDQN"

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
                    name    = env_name + "_" + alg_name,
                    net_dir = home_env + "Saved_Models",
                    \
                    gamma = 0.99, lr = 1e-4, grad_clip = 0,
                    \
                    input_dims = inp_space, n_actions = act_space,
                    depth = 3, width = 64,
                    activ = nn.PReLU(), noisy = True,
                    \
                    eps     = 1.0,
                    eps_min = 0.01,
                    eps_dec = 5e-5,
                    \
                    mem_size    = 100000, batch_size = 64,
                    target_sync = 1e-2,   freeze_up  = 500,
                    \
                    PER_on    = True, n_step   = 3,
                    PEReps    = 0.01, PERa     = 0.5,
                    PERbeta   = 0.4,  PERb_inc = 1e-7,
                    PERmax    = 1,
                    \
                    n_quantiles = 32
                    )

    if load_prev:
        agent.load_models()

    myUT.train_dqn_model( agent, env, render_on, test_mode, save_every,
                          draw_return, draw_interv, "quant" )


if __name__ == '__main__':
    main()
