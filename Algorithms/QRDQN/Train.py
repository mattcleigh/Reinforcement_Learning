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
from QRDQN import Agent
from Resources.Utils import score_plot
from Resources.Utils import quant_plot

def main():

    test_mode = False
    load_checkpoint = False

    render_on = True
    draw_return = False
    interval = 10
    best_score = 1500

    # env = Car_Env.MainEnv( rand_start = False )
    env = gym.make("Pong-ram-v0")

    ## We get the action and input shape from the environments themselves
    inp_space = list( env.reset().shape )
    act_space = env.action_space.n

    # print( env.unwrapped.get_action_meanings() )
    # print(input_space)
    # print(n_actions)
    # exit()

    agent = Agent(
                    name    = "cartpole_AI_QRDQN",
                    net_dir = home_env + "Saved_Models",
                    \
                    gamma = 0.99, lr = 1e-3,
                    \
                    input_dims = inp_space, n_actions = act_space,
                    depth = 3, width = 128,
                    activ = nn.PReLU(), noisy = True,
                    \
                    eps     = 1.0,
                    eps_min = 0.01,
                    eps_dec = 5e-5,
                    \
                    mem_size    = 500000, batch_size = 3,
                    target_sync = 100,    freeze_up   = 00,
                    \
                    PER_on    = True, n_step   = 3,
                    PEReps    = 0.01, PERa     = 0.5,
                    PERbeta   = 0.4,  PERb_inc = 1e-7,
                    PERmax = 1,
                    \
                    n_quantiles = 51
                    )

    if load_checkpoint:
        agent.load_models("_best")

    ## For plotting as it learns
    plt.ion()
    sp = score_plot("QRDQN")

    if draw_return:
        qp = quart_plot(0, 100)

    all_time = 0
    for ep in count():

        state = env.reset()
        ep_score = 0.0
        ep_loss  = 0.0

        for t in count():
            if render_on:
                env.render()

            action, dist = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            eps = agent.eps

            if not test_mode:
                agent.store_transition( state, action, reward, next_state, done )
                loss = agent.train()
                ep_loss += loss

            state = next_state

            ep_score += reward
            all_time += 1.0

            print( "Score = {:.7}     \r".format( ep_score ), end="" )
            sys.stdout.flush()

            if draw_return and all_time%interval==0:
                qp.update(dist)

            if not test_mode:
                if all_time>=10000 and all_time%10000==0:
                    agent.save_models()
                if ep_score==best_score:
                    agent.save_models("_best")

            if done: break

        ep_loss  /= (t+1)

        print( "Episode {}: Reward = {:.7}, Loss = {:4.3f}, Eps = {:4.3f}, Episode Time = {}, Total Time = {}".format( \
                                         ep, ep_score, ep_loss, eps, t+1, all_time ) )

        sp.update(ep_score)


if __name__ == '__main__':
    main()
