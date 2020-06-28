import sys
home_env = '../../../Reinforcement_Learning/'
sys.path.append(home_env)

from Resources import MemoryMethods as mm
from Resources.Utils import score_plot
from Environments import Car_Env

import gym
import os
import time
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict

class Worker:
    """ This worker object contains its own environment and
        records its own experience.
        It is linked to the central agent.
    """
    def __init__(self, env_name, cen_agent, n_frames):

        if env_name=="car":
            self.env = Car_Env.MainEnv( rand_start = True )
        else:
            self.env = gym.make(env_name)

        self.state = self.env.reset()
        self.cen_agent = cen_agent
        self.n_frames = n_frames
        self.ep_score = 0.0

    def fill_batch(self):

        for _ in range(self.n_frames):
            action = self.cen_agent.choose_action(self.state)
            n_s, rew, done, info = self.env.step(action)
            self.cen_agent.store_transition( self.state, action, rew, n_s, done )
            self.state = n_s
            self.ep_score += rew

            if done:
                self.state = self.env.reset()
                self.cen_agent.sp.update(self.ep_score)
                self.ep_score = 0.0



class ActorCriticMLP(nn.Module):
    """ A simple and configurable multilayer perceptron.
        An actor-critic method usually includes one network each.
        However, feature extraction usually requires the same tools.
        Thus, they share the same base layer.
    """
    def __init__(self, name, chpt_dir,
                       input_dims, n_actions,
                       depth, width, activ ):
        super(ActorCriticMLP, self).__init__()

        ## Defining the network features
        self.name       = name
        self.chpt_dir   = chpt_dir
        self.chpt_file  = os.path.join(self.chpt_dir, self.name)
        self.input_dims = input_dims
        self.n_actions  = n_actions

        ## Defining the base (shared) layer structure
        layers = []
        for l_num in range(1, depth+1):
            inpt = input_dims[0] if l_num == 1 else width
            layers.append(( "base_lin_{}".format(l_num), nn.Linear(inpt, width) ))
            layers.append(( "base_act_{}".format(l_num), activ ))
        self.base_stream = nn.Sequential(OrderedDict(layers))

        ## Defining the actor network, returns the policy (softmax)
        self.actor_stream = nn.Sequential(OrderedDict([
            ( "actor_lin_1",   nn.Linear(width, width) ),
            ( "actor_act_1",   activ ),
            ( "actor_lin_out", nn.Linear(width, n_actions) ),
            ( "actor_act_out", nn.Softmax(dim=-1) ),
        ]))

        ## Defining the critic network, returns the state value function
        self.critic_stream = nn.Sequential(OrderedDict([
            ( "critic_lin_1",   nn.Linear(width, width) ),
            ( "critic_act_1",   activ ),
            ( "critic_lin_out", nn.Linear(width, 1) ),
        ]))

        ## Moving the network to the device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        shared_out = self.base_stream(state)
        policy = self.actor_stream(shared_out)
        value  = self.critic_stream(shared_out)
        return policy, value

    def get_value(self, state):
        shared_out = self.base_stream(state)
        value = self.critic_stream(shared_out)
        return value

    def get_policy(self, state):
        shared_out = self.base_stream(state)
        policy = self.actor_stream(shared_out)
        return policy

    def save_checkpoint(self, flag=""):
        print("... saving network checkpoint ..." )
        T.save(self.state_dict(), self.chpt_file+flag)

    def load_checkpoint(self, flag=""):
        print("... loading network checkpoint ..." )
        self.load_state_dict(T.load(self.chpt_file+flag))


class Agent(object):
    def __init__(self,
                 name,
                 net_dir,
                 \
                 gamma, lr,
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
        self.actor_critic = ActorCriticMLP( self.name + "_ac_networks", net_dir,
                                            input_dims, n_actions,
                                            depth, width, activ )

        ## The gradient descent algorithm and loss function used to train the policy network
        self.optimiser = optim.Adam( self.actor_critic.parameters(), lr = lr )

        ## We create the memory to hold the update for each batch
        self.memory = mm.SmallMemory( n_workers*n_frames, input_dims )
        self.memory.reset()

        ## We now initiate the list of workers
        self.workers = [ Worker(env_name, self, n_frames) for _ in range(n_workers) ]

        ## We also initiate the graph, which is an agent attribute as it is called by all workers
        plt.ion()
        self.sp = score_plot("A2c")

    def save_model(self, flag=""):
        self.actor_critic.save_checkpoint(flag)

    def load_model(self, flag=""):
        self.actor_critic.load_checkpoint(flag)

    def store_transition(self, state, action, reward, next_state, done):
        ## Interface to memory, so no outside class directly calls it
        self.memory.store_transition(state, action, reward, next_state, done)

    def choose_action(self, state):

        ## First we convert the state observation into a tensor
        state_tensor = T.tensor( [state], device=self.actor_critic.device, dtype=T.float32 )

        ## We then calculate the probabilities of taking each action
        policy = self.actor_critic.get_policy( state_tensor )

        ## To sample an action using these probabilities we use the distributions package
        action_dist   = T.distributions.Categorical(policy)
        chosen_action = action_dist.sample()

        return chosen_action.item()

    def train(self):

        ## We zero out the gradients, as required for each pyrotch train loop
        self.optimiser.zero_grad()

        ## First we have each worker fill up its portion of the batch
        self.memory.reset()
        for worker in self.workers:
            worker.fill_batch()

        ## We need to convert all of these arrays to pytorch tensors
        states      = T.tensor(self.memory.states).to(self.actor_critic.device)
        actions     = T.tensor(self.memory.actions).to(self.actor_critic.device)
        rewards     = T.tensor(self.memory.rewards).to(self.actor_critic.device).reshape(-1, 1)
        next_states = T.tensor(self.memory.next_states).to(self.actor_critic.device)
        dones       = T.tensor(self.memory.dones).to(self.actor_critic.device).reshape(-1, 1)

        ## We need both the value and policy based on the current states
        policy, state_values = self.actor_critic(states)

        ## Now we use the critic to get the value of the next states
        next_state_values = self.actor_critic.get_value(next_states)

        ## We use the td_error as an estimator of the advantage value and the critic loss
        td_targets  = rewards + self.gamma * next_state_values * (~dones)
        td_errors   = td_targets - state_values
        critic_loss = td_errors.pow(2).mean()

        ## To get the policy loss we start with the distribution over the actions
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
        nn.utils.clip_grad_norm_( self.actor_critic.parameters(), 0.5 )
        self.optimiser.step()

        return loss.item()
