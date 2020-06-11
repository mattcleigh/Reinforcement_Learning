import sys
sys.path.append('/home/matthew/Documents/Reinforcement_Learning/')

import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

from Environments import Car_Env
from ND3QN import Agent
from RLResources.Utils import score_plot
from RLResources.Utils import value_plot

def main():

    load_checkpoint = False
    draw_return = False
    saving_on = False

    env = Car_Env.MainEnv()
    # env = gym.make("CartPole-v0")

    agent = Agent( name = "car_AI",
                   gamma       = 0.99,     lr = 5e-5,
                   input_dims  = [28],     n_actions  = 12,
                   mem_size    = 1000000,  batch_size = 64,
                   target_sync = 5e-4,     freeze_up  = 2000,
                   PEReps      = 0.01,     PERa       = 0.5,
                   PERbeta     = 0.4,      PERb_inc   = 1e-6,
                   PERmax_td   = 5,        n_step     = 3,
                   net_dir = "/home/matthew/Documents/Reinforcement_Learning/ND3QN/Saved_Binaries" )

    if load_checkpoint:
        agent.load_models()

    ## For plotting the scores as it learns
    plt.ion()
    sp = score_plot()

    if draw_return:
        rp = value_plot()

    all_time = 0
    for ep in count():

        state = env.reset()

        ep_score = 0
        ep_loss  = 0
        ep_error = 0

        for t in count():
            # env.render()

            action, value = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition( state, action, reward, next_state, done )
            loss, error = agent.train()

            state = next_state

            ep_score += reward
            ep_loss  += loss
            ep_error += error
            all_time += 1

            if draw_return:
                rp.update(dist)

            if saving_on and all_time>=10000 and all_time%10000==0:
                agent.save_models()

            if saving_on and ep_score==1000:
                agent.save_models("_best")

            if done: break


        ep_loss  /= (t+1)
        ep_error /= (t+1)

        print( "Episode {}: Reward = {}, Loss = {:4.3f}, Error = {:4.3f}, Episode Time = {}, Total Time = {}".format( \
                                         ep, ep_score, ep_loss, ep_error, t+1, all_time ) )

        sp.update(ep_score)



if __name__ == '__main__':
    main()
















