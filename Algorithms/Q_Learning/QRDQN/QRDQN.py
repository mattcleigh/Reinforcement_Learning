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

class QRDuelMLP(nn.Module):
    """ A simple and configurable multilayer perceptron.
        This is a distributional arcitecture for QR.
        This is a dueling network and contains seperate streams
        for value and advantage evaluation.
        The seperate streams can be equipped with noisy layers.
    """
    def __init__(self, name, chpt_dir,
                       input_dims, n_actions,
                       depth, width, activ,
                       noisy,
                       n_quantiles ):
        super(QRDuelMLP, self).__init__()

        ## Defining the network features
        self.name       = name
        self.chpt_dir   = chpt_dir
        self.chpt_file  = os.path.join(self.chpt_dir, self.name)
        self.input_dims = input_dims
        self.n_actions  = n_actions

        ## The QR distributional RL parameters
        self.n_quantiles = n_quantiles

        ## Defining the base and dueling layer structures
        self.base_stream = myNN.mlp_creator( "base", n_in=input_dims[0], d=depth, w=width, act_h=activ )
        self.V_stream = myNN.mlp_creator( "V", n_in=width, n_out=n_quantiles, w=width, act_h=activ, nsy=noisy )
        self.A_stream = myNN.mlp_creator( "A", n_in=width, n_out=n_actions*n_quantiles, w=width, act_h=activ, nsy=noisy )

        ## Moving the network to the device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, state):
        ## This is a network for the QR algorithm
        ## So the output is a matrix AxN
            ## Each row is an action
            ## Each column is the location of a Q dist quantile

        shared_out = self.base_stream(state)
        V = self.V_stream(shared_out).view(-1, 1, self.n_quantiles)
        A = self.A_stream(shared_out).view(-1, self.n_actions, self.n_quantiles)
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
                 \
                 n_quantiles,
                 ):

        ## Setting all class variables
        self.__dict__.update(locals())
        self.learn_step_counter = 0
        self.n_gamma = self.gamma ** self.n_step

        ## The policy and target networks
        self.policy_net = QRDuelMLP( self.name + "_policy_network", net_dir,
                                     input_dims, n_actions, depth, width, activ,
                                     noisy, n_quantiles )
        self.target_net = QRDuelMLP( self.name + "_target_network", net_dir,
                                     input_dims, n_actions, depth, width, activ,
                                     noisy, n_quantiles )
        self.target_net.load_state_dict( self.policy_net.state_dict() )

        ## The gradient descent algorithm used to train the policy network
        self.optimiser = optim.Adam( self.policy_net.parameters(), lr = lr )
        self.huber_fn  = lambda x: T.where( x.abs() < 1, 0.5 * x.pow(2), (x.abs() - 0.5) )

        ## The agent memory
        self.memory = myUT.memory_creator( PER_on, n_step, gamma, mem_size,
                                           input_dims, PEReps, PERa,
                                           PERbeta, PERb_inc, PERmax )


    def choose_action(self, state):

        ## Act completly randomly for the first x frames
        if self.memory.mem_cntr < self.freeze_up:
            action = rd.randint(self.n_actions)
            act_dist = np.zeros( self.n_quantiles )

        ## If there are no noisy layers then we must do e-greedy
        elif not self.noisy and rd.random() < self.eps:
                action = rd.randint(self.n_actions)
                act_dist = np.zeros( self.n_quantiles )
                self.eps = max( self.eps - self.eps_dec, self.eps_min )

        ## Then act purely greedily
        else:
            with T.no_grad():
                state_tensor = T.tensor( [state], device=self.target_net.device, dtype=T.float32 )
                dist = self.policy_net(state_tensor)
                Q_values = T.sum( dist, dim=-1 )
                action = T.argmax(Q_values, dim=1).item()
                act_dist = dist[0][action].cpu().numpy()

        return action, act_dist


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
        rewards     = T.tensor(rewards).to(self.policy_net.device).reshape(-1, 1)
        next_states = T.tensor(next_states).to(self.policy_net.device)
        dones       = T.tensor(dones).to(self.policy_net.device).reshape(-1, 1)
        is_weights  = T.tensor(is_weights).to(self.policy_net.device)

        ## We use the range of up to batch_size just for indexing methods
        batch_idxes = list(range(self.batch_size))

        ## To increase the speed of this step we do it without keeping track of gradients
        with T.no_grad():

            ## First we need the next state distribution using the policy network for double Q learning
            pol_dist_next = self.policy_net(next_states)

            ## We then find the Q-values of the actions by summing over the supports
            pol_Q_next = T.sum( pol_dist_next, dim=-1 )

            ## Can then determine the optimum actions
            next_actions = T.argmax(pol_Q_next, dim=1)

            ## We now use the target network to get the distributions of these actions
            tar_dist_next = self.target_net(next_states)[batch_idxes, next_actions]

            ## We can then find the new supports using the distributional Bellman Equation
            td_target_dist = rewards + self.n_gamma * tar_dist_next * (~dones)
            td_target_dist = td_target_dist.detach()

        ## Now we want to track gradients using the policy network
        pol_dist = self.policy_net(states)[batch_idxes, actions]

        taus = T.arange( start=1, end=self.n_quantiles+1, dtype=T.float32, device=self.policy_net.device )
        taus = ( 2 * ( taus - 1 ) + 1 ) / ( 2 * self.n_quantiles )

        ## To create the difference tensor for each sample in batch various unsqueezes are needed
        dist_diff = td_target_dist.unsqueeze(-1) - pol_dist.unsqueeze(-1).transpose(1,2)

        ## We then find the loss function using the QR equation
        QRloss = self.huber_fn(dist_diff) * (taus - (dist_diff.detach()<0).float()).abs()

        ## We then need to find the mean along the batch dimension, so we get the loss for each sample
        QRloss = QRloss.sum(dim=-1).mean(dim=-1)

        ## Use this loss as new errors to be used in PER and update the replay
        if self.PER_on:
            new_errors = QRloss.detach().cpu().numpy().squeeze()
            self.memory.batch_update(indices, new_errors)

        ## Now we use the loss for graidient desc, applying is weights if using PER
        loss = QRloss
        if self.PER_on:
            loss = loss * is_weights
        loss = loss.mean()
        loss.backward()

        ## We might want to clip the gradient before performing SGD
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_( self.policy_net.parameters(), self.grad_clip )
        self.optimiser.step()

        self.learn_step_counter += 1

        return loss.item()
