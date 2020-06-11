import sys
sys.path.append('/home/matthew/Documents/Reinforcement_Learning/')

import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

from Environments import Car_Env
from Rainbow import Agent
from RLResources.Utils import score_plot
from RLResources.Utils import distribution_plot

def main():

    load_checkpoint = True
    draw_return = False
    saving_on = False

    env = Car_Env.MainEnv()
    # env = gym.make("LunarLander-v2")

    agent = Agent( name = "car_AI",
                   gamma       = 0.99,     lr = 5e-6,
                   input_dims  = [28],     n_actions  = 12,
                   mem_size    = 1000000,  batch_size = 128,
                   target_sync = 1e-3,     freeze_up  = 0000,
                   PEReps      = 0.01,     PERa       = 0.5,
                   PERbeta     = 0.4,      PERb_inc   = 2e-7,
                   PERmax_td   = 10,       n_step     = 3,
                   n_atoms     = 51,       sup_range  = [-1, 15],
                   net_dir = "/home/matthew/Documents/Reinforcement_Learning/Rainbow/Saved_Binaries" )

    if load_checkpoint:
        agent.load_models("_best")

    ## For plotting as it learns
    plt.ion()
    sp = score_plot()

    if draw_return:
        dp = distribution_plot( agent.vmin, agent.vmax, agent.n_atoms)

    all_time = 0
    for ep in count():

        state = env.reset()
        ep_score = 0
        ep_loss  = 0
        ep_error = 0

        for t in count():
            env.render()

            action, dist = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            # agent.store_transition( state, action, reward, next_state, done )
            # loss, error = agent.train()

            state = next_state

            ep_score += 1.0 #reward
            ep_loss  += 1.0 #loss
            ep_error += 1.0 #error
            all_time += 1.0

            if draw_return:
                dp.update(dist)

            if saving_on and all_time>=10000 and all_time%10000==0:
                agent.save_models()

            if saving_on and ep_score==2000:
                agent.save_models("_best")

            if done: break

        ep_loss  /= (t+1)
        ep_error /= (t+1)

        print( "Episode {}: Reward = {:.7}, Loss = {:4.3f}, Error = {:4.3f}, Episode Time = {}, Total Time = {}".format( \
                                         ep, ep_score, ep_loss, ep_error, t+1, all_time ) )

        sp.update(ep_score)


if __name__ == '__main__':
    main()
















