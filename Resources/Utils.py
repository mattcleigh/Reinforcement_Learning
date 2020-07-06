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

class Vectorised_Worker:
    """ This worker object contains its own environment and
        records its own experience.
        It is linked to the central agent.
    """
    def __init__(self, cen_agent, n_workers, env_name, n_frames, gamma):

        ## The standard class attributes
        self.cen_agent = cen_agent
        self.n_workers = n_workers
        self.n_frames = n_frames
        self.gamma = gamma

        ## The list of environments, current states, and scores
        self.envs = [ gym.make(env_name) for _ in range(n_workers) ]
        self.states = [ env.reset() for env in self.envs ]
        self.scores = np.zeros(n_workers)

        ## We check if we are in ram mode, which requires state pre-scaling
        self.ram_mode = False
        if len(self.states[0])==128:
            print("Ram input detected, will be applying pre-scaling:")
            print("----->  X / 255 - 0.5")
            for i in range(n_workers):
                self.states[i] = self.states[i] / 255 - 0.5
            self.ram_mode = True

    def fill_batch(self, render_on):

        ## The batch memory of this instance
        states  = [ [] for _ in range(self.n_workers) ]
        actions = [ [] for _ in range(self.n_workers) ]
        rewards = [ [] for _ in range(self.n_workers) ]
        dones   = [ [] for _ in range(self.n_workers) ]

        ## We now iterate through all frames
        for _ in range(self.n_frames):

            ## We call the network once to choose an action for all envs
            chosen_actions = self.cen_agent.vector_choose_action(self.states)

            ## We now need to manually step once through each environment
            for i in range(self.n_workers):

                ## Render only the first environment
                if i==0 and render_on:
                    self.envs[i].render()

                ## Receive new observations
                next_state, reward, done, info = self.envs[i].step(chosen_actions[i])

                ## Ram mode pre-scaling
                if self.ram_mode:
                    next_state = next_state / 255 - 0.5

                ## Storing the particulars of this environment in its own array
                states[i].append(self.states[i])
                actions[i].append(chosen_actions[i])
                rewards[i].append(reward)
                dones[i].append(done)

                ## Updating the scores and states
                self.scores[i] += reward
                self.states[i] = next_state

                ## If this particular env has ended
                if done:
                    self.states[i] = self.envs[i].reset()
                    self.cen_agent.sp.update(self.scores[i])
                    self.scores[i] = 0.0

                    ## Ram mode pre-scaling
                    if self.ram_mode:
                        self.states[i] = self.states[i] / 255 - 0.5

        ## Now that the batch is full we try calculate the n_step returns
        values = [ [] for _ in range(self.n_workers) ]

        ## We do this seperately for each individual environment
        for i in range(self.n_workers):

            ## The next value after our final action is 0 unless the episode continues
            next_value = 0
            if not dones[i][-1]:
                state_tensor = T.tensor( [states[i][-1]], device=self.cen_agent.actor_critic.device, dtype=T.float32 )
                next_value = self.cen_agent.actor_critic.get_value(state_tensor).item()

            ## From there we begin discounting and working backward reseting at each ep lim
            for t in reversed(range(self.n_frames)):
                if not dones[i][t]:
                    next_value = rewards[i][t] + next_value * self.gamma
                else:
                    next_value = rewards[i][t]
                values[i].append(next_value)

            values[i].reverse()

        ## Now that all the arrays are filled, we flatten them and return them
        states_fl  = [ item for sublist in states  for item in sublist ]
        actions_fl = [ item for sublist in actions for item in sublist ]
        values_fl  = [ item for sublist in values  for item in sublist ]

        return states_fl, actions_fl, values_fl


class Worker:
    """ This class is no longer in use since switching to vectorised
        workers (above) which acts faster on the gpu. It is kept for legacy purposes,
        incase I want to switch to multithreaded methods.

        This worker object contains its own environment and
        records its own experience.
        It is linked to the central agent.
    """
    def __init__(self, cen_agent, env_name, n_frames, gamma):

        if env_name=="car":
            self.env = Car_Env.MainEnv( rand_start = True )
        else:
            self.env = gym.make(env_name)

        ## We check if we are in ram mode, which requires state pre-scaling
        self.state = self.env.reset()
        self.ram_mode = False
        if len(self.state)==128:
            print("Ram input detected, will be applying pre-scaling:")
            print("----->  X / 255 - 0.5")
            self.state = self.state / 255 - 0.5
            self.ram_mode = True

        self.cen_agent = cen_agent
        self.n_frames = n_frames
        self.gamma = gamma
        self.ep_score = 0.0

    def fill_batch(self):
        states, actions, rewards, dones = [], [], [], []

        for _ in range(self.n_frames):
            action = self.cen_agent.choose_action(self.state)
            next_state, reward, done, info = self.env.step(action)

            if self.ram_mode:
                next_state = next_state / 255 - 0.5

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
                if self.ram_mode:
                    self.state = self.state / 255 - 0.5


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

    ## We check if we are in ram mode, which requires state pre-scaling
    ram_mode = False
    if len(env.reset())==128:
        print("\nRam input detected, will be applying pre-scaling:")
        print("----->  X / 255 - 0.5\n")
        ram_mode = True

    ## Episode loop
    all_time = 0
    for ep in count():

        ## Resetting the environment and episode stats
        state = env.reset()
        ep_score = 0.0
        ep_loss  = 0.0

        ## We check if pre-scaling needs to be done
        if ram_mode:
            state = state / 255 - 0.5

        ## Running through an episode
        for t in count():

            ## We visualise the environment if we want
            if render_on:
                env.render()

            ## The agent chooses an action
            action, value = agent.choose_action(state)

            ## The environment evolves wrt chosen action
            next_state, reward, done, info = env.step(action)

            ## We check if pre-scaling needs to be done
            if ram_mode:
                next_state = next_state / 255 - 0.5

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


def train_ac_model(agent, env, render_on, test_mode, save_every):
    loss = 0
    ## Main training loop
    for t in count():

        states, actions, values = agent.vector_step(render_on)

        if not test_mode:
            loss = agent.train(states, actions, values)


        if not test_mode:
            if t>=20 and t%20==0:
                agent.save_models()

        print( "Iteration = {}, Loss = {:4.3f}        \r".format( t, loss ), end="" )
        sys.stdout.flush()
