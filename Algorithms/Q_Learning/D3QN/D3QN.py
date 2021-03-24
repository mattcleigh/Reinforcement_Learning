from Resources import Networks as myNN
from Resources import Utils as myUT

import os
import time
import numpy as np
import numpy.random as rd

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict

class DuelMLP(nn.Module):
    """ A basic arcitecture used for the D3QN algorithm.
        A dueling network, containing seperate streams
        for value and advantage evaluation.
        The seperate streams can be equipped with noisy layers.
    """
    def __init__(self, name, chpt_dir,
                       input_dims, n_actions,
                       depth, width, activ,
                       noisy ):
        super(DuelMLP, self).__init__()

        ## Defining the network features
        self.name       = name
        self.chpt_dir   = chpt_dir
        self.chpt_file  = os.path.join(self.chpt_dir, self.name)
        self.input_dims = input_dims
        self.n_actions  = n_actions

        ## Defining the base and dueling layer structures
        self.base_stream = myNN.mlp_creator( "base", n_in=input_dims[0], d=depth, w=width, act_h=activ )
        self.V_stream = myNN.mlp_creator( "V", n_in=width, n_out=1, w=width, act_h=activ, nsy=noisy )
        self.A_stream = myNN.mlp_creator( "A", n_in=width, n_out=n_actions, w=width, act_h=activ, nsy=noisy )

        ## Moving the network to the device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        shared_out = self.base_stream(state)
        V = self.V_stream(shared_out)
        A = self.A_stream(shared_out)
        Q = V + A - A.mean( dim=1, keepdim=True)
        return Q

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
                 gamma, lr, grad_clip,
                 \
                 input_dims,  n_actions,
                 depth, width,
                 activ, noisy,
                 \
                 eps, eps_min, eps_dec,
                 \
                 mem_size,    batch_size,
                 target_sync, freeze_up,
                 \
                 PER_on,      n_step,
                 PEReps,      PERa,
                 PERbeta,     PERb_inc,
                 PERmax,
                 ):

        ## Setting all class variables
        self.__dict__.update(locals())
        self.learn_step_counter = 0
        self.n_gamma = self.gamma ** self.n_step

        ## The policy and target networks
        self.policy_net = DuelMLP( self.name + "_policy_network", net_dir,
                                   input_dims, n_actions, depth, width, activ, noisy )
        self.target_net = DuelMLP( self.name + "_target_network", net_dir,
                                   input_dims, n_actions, depth, width, activ, noisy )
        self.target_net.load_state_dict( self.policy_net.state_dict() )

        ## The gradient descent algorithm and loss function used to train the policy network
        self.optimiser = optim.Adam( self.policy_net.parameters(), lr = lr )
        self.loss_fn = nn.SmoothL1Loss( reduction = "none" )

        ## The agent memory
        self.memory = myUT.memory_creator( PER_on, n_step, gamma, mem_size,
                                           input_dims, PEReps, PERa,
                                           PERbeta, PERb_inc, PERmax )

    def choose_action(self, state):

        ## Act completly randomly for the first x frames
        if self.memory.mem_cntr < self.freeze_up:
            action = rd.randint(self.n_actions)
            act_value = 0

        ## If there are no noisy layers then we must do e-greedy
        elif not self.noisy and rd.random() < self.eps:
                action = rd.randint(self.n_actions)
                act_value = 0
                self.eps = max( self.eps - self.eps_dec, self.eps_min )

        ## Then act purely greedily
        else:
            with T.no_grad():
                state_tensor = T.tensor( [state], device=self.target_net.device, dtype=T.float32 )
                Q_values = self.policy_net(state_tensor)
                action = T.argmax(Q_values).item()
                act_value = Q_values[0][action].cpu().numpy()

        return action, act_value

    def store_transition(self, state, action, reward, next_state, done):
        ## Interface to memory, so no outside class directly calls it
        self.memory.store_transition(state, action, reward, next_state, done)

    def sync_target_network(self):

        ## If we are doing soft updates
        if self.target_sync < 1:
            with T.no_grad():
                for tp, pp in zip( self.target_net.parameters(), self.policy_net.parameters() ):
                    tp.data.copy_( self.target_sync * pp.data + ( 1.0 - self.target_sync ) * tp.data )

        ## If we are doing hard updates
        else:
            if self.learn_step_counter % self.target_sync == 0:
                self.target_net.load_state_dict( self.policy_net.state_dict() )

    def save_models(self, flag=""):
        self.policy_net.save_checkpoint(flag)
        self.target_net.save_checkpoint(flag)

    def load_models(self, flag=""):
        self.policy_net.load_checkpoint(flag)
        self.target_net.load_checkpoint(flag)

    def train(self):

        ## We dont train until the memory is at least one batch_size
        if self.memory.mem_cntr < max(self.batch_size, self.freeze_up):
            return 0

        ## We check if the target network needs to be replaced
        self.sync_target_network()

        ## We zero out the gradients, as required for each pyrotch train loop
        self.optimiser.zero_grad()

        ## Collect the batch
        states, actions, rewards, next_states, dones, is_weights, indices = self.memory.sample_memory(self.batch_size)

        ## We need to convert all of these arrays to pytorch tensors
        states      = T.tensor(states).to(self.policy_net.device)
        actions     = T.tensor(actions).to(self.policy_net.device)
        rewards     = T.tensor(rewards).to(self.policy_net.device)
        next_states = T.tensor(next_states).to(self.policy_net.device)
        dones       = T.tensor(dones).to(self.policy_net.device)
        is_weights  = T.tensor(is_weights).to(self.policy_net.device)

        ## We use the range of up to batch_size just for indexing methods
        batch_idxes = list(range(self.batch_size))

        ## To increase the speed of this step we do it without keeping track of gradients
        with T.no_grad():

            ## First we find the Q-values of the next states
            pol_Q_next = self.policy_net(next_states)

            ## Can then determine the optimum actions
            next_actions = T.argmax(pol_Q_next, dim=1)

            ## We now use the target network to get the values of these actions
            tar_Q_next = self.target_net(next_states)[batch_idxes, next_actions]

            ## Calculate the target values based on the Bellman Equation
            td_target = rewards + self.n_gamma * tar_Q_next * (~dones)
            td_target = td_target.detach()

        ## Now we calculate the network estimates for the state values
        pol_Q = self.policy_net(states)[batch_idxes, actions]

        ## Calculate the TD-Errors to be used in PER and update the replay
        if self.PER_on:
            new_errors = T.abs(pol_Q - td_target).detach().cpu().numpy().squeeze()
            self.memory.batch_update(indices, new_errors)

        ## Now we use the loss for graidient desc, applying is weights if using PER
        loss = self.loss_fn( pol_Q, td_target )
        if self.PER_on:
            loss = loss * is_weights.unsqueeze(1)
        loss = loss.mean()
        loss.backward()

        ## We might want to clip the gradient before performing SGD
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_( self.policy_net.parameters(), self.grad_clip )
        self.optimiser.step()

        self.learn_step_counter += 1

        return loss.item()
