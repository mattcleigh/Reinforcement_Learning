import sys
home_env = '../../../Reinforcement_Learning/'
sys.path.append(home_env)

import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from itertools import count

import torch as T
import torch.nn as nn

from Environments import Car_Env
from Resources import Plotting as myPT

import torch as T

class Worker:
    """ This worker object contains its own environment and
        records its own experience.
        It is linked to the central agent.
    """
    def __init__(self, cen_agent, env_name, n_frames, gamma):

        if env_name=="car":
            self.env = Car_Env.MainEnv( rand_start = True )
        else:
            self.env = gym.make(env_name)

        self.state = self.env.reset()
        self.cen_agent = cen_agent
        self.n_frames = n_frames
        self.gamma = gamma
        self.ep_score = 0.0

    def fill_batch(self):
        states, actions, rewards, dones = [], [], [], []

        for _ in range(self.n_frames):
            action = self.cen_agent.choose_action(self.state)
            next_state, reward, done, info = self.env.step(action)

            states.append(self.state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            self.ep_score += reward
            self.state = next_state

            if done:
                self.state = self.env.reset()
                self.cen_agent.sp.update(self.ep_score)
                self.ep_score = 0.0

        ## Now that the batch is full we try calculate the n_step returns
        values = []

        ## The next value after our final action is 0 unless the episode continues
        next_value = 0
        if not dones[-1]:
            state_tensor = T.tensor( [states[-1]], device=self.cen_agent.actor_critic.device, dtype=T.float32 )
            next_value = self.cen_agent.actor_critic.get_value(state_tensor).item()

        ## From there we begin discounting and working backward reseting at each ep lim
        for i in reversed(range(self.n_frames)):
            if not dones[i]:
                next_value = rewards[i] + next_value * self.gamma
            else:
                next_value = rewards[i]
            values.append(next_value)

        values.reverse()

        ## Now we iterate through the new batch and store it to memory
        for s, a, v in zip(states, actions, values):
            self.cen_agent.store_transition( s, a, v )


def train_dqn_model( agent, env, render_on, test_mode, save_every,
                     draw_return, draw_interv, ret_type="" ):

    ## For plotting as it learns
    plt.ion()
    sp = myPT.score_plot(agent.name)
    if draw_return:
        if  ret_type=="val":
            vp = myPT.value_plot()
        elif ret_type=="dist":
            vp = myPT.dist_plot( agent.vmin, agent.vmax, agent.n_atoms)
        elif ret_type=="quant":
            vp = myPT.quant_plot()

    ## Episode loop
    all_time = 0
    for ep in count():

        ## Resetting the environment and episode stats
        state = env.reset()
        ep_score = 0.0
        ep_loss  = 0.0

        ## Running through an episode
        for t in count():

            ## We visualise the environment if we want
            if render_on:
                env.render()

            ## The agent chooses an action
            action, value = agent.choose_action(state)

            ## The environment evolves wrt chosen action
            next_state, reward, done, info = env.step(action)

            ## Storing the transition and training the model
            if not test_mode:
                agent.store_transition( state, action, reward, next_state, done )
                loss = agent.train()
                ep_loss += loss

            ## Replacing the state
            state = next_state

            ## Updating episode stats
            ep_score += reward
            all_time += 1.0
            eps = agent.eps

            ## Printing running episode score
            print( "Score = {:.7}     \r".format( ep_score ), end="" )
            sys.stdout.flush()

            ## Drawing the modelled return
            if draw_return and all_time%draw_interv==0:
                vp.update(value)

            ## Saving the models
            if not test_mode and all_time%save_every==0:
                agent.save_models()

            ## Check if episode has concluded
            if done:
                break

        ## Prining and plotting the completed episode stats
        ep_loss /= (t+1)
        sp.update(ep_score)
        print( "Episode {}: Reward = {:.7}, Loss = {:4.3f}, Eps = {:4.3f}, Episode Time = {}, Total Time = {}".format( \
                                         ep, ep_score, ep_loss, eps, t+1, all_time ) )
