import sys
home_env = '../../../Reinforcement_Learning/'
sys.path.append(home_env)

from Resources import Networks as myNN
from Resources import Plotting as myPT
from Resources import Utils as myUT
from Environments import Car_Env

import gym
import os
import time
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Agent(object):
    def __init__(self,
                 name,
                 net_dir,
                 \
                 gamma, lr, grad_clip,
                 \
                 input_dims,  n_actions,
                 depth, width, activ,
                 \
                 env_name,
                 n_workers, n_frames,
                 vf_coef, ent_coef,
                 ):

        ## Setting all class variables
        self.__dict__.update(locals())
        self.learn_step_counter = 0

        ## The actor and critic networks are initialised
        self.actor_critic = myNN.ActorCriticMLP( self.name + "_ac_networks", net_dir,
                                            input_dims, n_actions,
                                            depth, width, activ )

        ## The gradient descent algorithm and loss function used to train the policy network
        self.optimiser = optim.Adam( self.actor_critic.parameters(), lr = lr )
        self.loss_fn = nn.SmoothL1Loss()

        ## We create the memory to hold the update for each batch
        self.memory = myMM.SmallMemory( n_workers*n_frames, input_dims )
        self.memory.reset()

        ## We now initiate the vectorised worker environment
        self.vec_workers = myUT.Vectorised_Worker(self, n_workers, env_name, n_frames, gamma)

        ## We also initiate the graph, which is an agent attribute as it is called by all workers
        plt.ion()
        self.sp = myPT.score_plot(self.name)

    def save_models(self, flag=""):
        self.actor_critic.save_checkpoint(flag)

    def load_models(self, flag=""):
        self.actor_critic.load_checkpoint(flag)

    def vector_step(self, render_on):
        return self.vec_workers.fill_batch(render_on)

    def vector_choose_action(self, states):

        with T.no_grad():
            ## First we convert the many states into a tensor
            state_tensor = T.tensor( states, device=self.actor_critic.device, dtype=T.float32 )

            ## We then calculate the probabilities of taking each action
            policy = self.actor_critic.get_policy( state_tensor )

            ## We then sample the policies to get the action taken in each env
            action_dists   = T.distributions.Categorical(policy)
            chosen_actions = action_dists.sample()

            return chosen_actions.cpu().detach().numpy()

    def train(self, states, actions, values):

        ## We get the new batch and convert to tensors
        states  = T.tensor(states, dtype=T.float32, device=self.actor_critic.device)
        actions = T.tensor(actions, dtype=T.int64, device=self.actor_critic.device)
        values  = T.tensor(values, dtype=T.float32, device=self.actor_critic.device).reshape(-1,1)

        ## We zero out the gradients, as required for each pyrotch train loop
        self.optimiser.zero_grad()

        ## We start by calculating the critic/value loss
        ## We need both the value and policy based on the current states
        policy, state_values = self.actor_critic(states)

        ## We use the td_error as an estimator of the advantage value and the critic loss
        td_errors   = values - state_values
        critic_loss = self.loss_fn(state_values, values)

        ## Now we move onto the actor/policy loss
        ## We start with the distribution over the actions taken
        action_dist = T.distributions.Categorical(policy)

        ## We now want to calculate the log_probs of the chosen actions
        log_probs = action_dist.log_prob(actions).view(-1, 1)

        ## We calculate the loss of the actor (negative for SGA)
        actor_loss  = -(log_probs * td_errors.detach()).mean()

        ## We finally use the distribution to get the regularising entropy
        entropy = action_dist.entropy().mean()

        ## We do a single update step using the sum of losses (equivalent to two steps)
        loss = actor_loss + ( self.vf_coef * critic_loss ) - ( self.ent_coef * entropy )
        loss.backward()

        ## We might want to clip the gradient before performing SGD
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_( self.actor_critic.parameters(), self.grad_clip )
        self.optimiser.step()

        self.learn_step_counter += 1

        return loss.item()
