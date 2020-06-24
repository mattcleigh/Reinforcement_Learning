import sys
sys.path.append('/home/matthew/Documents/Reinforcement_Learning/')

from Resources import Layers as ll
from Resources import MemoryMethods as MM

import os
import time
import numpy as np
import numpy.random as rd

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict

class C51DuelMLP(nn.Module):
    """ A simple and configurable multilayer perceptron.
        This is a distributional arcitecture for the C51 algoritm.
        This is a dueling network and contains seperate streams
        for value and advantage evaluation.
        The seperate streams can be equipped with noisy layers.
    """
    def __init__(self, name, chpt_dir,
                       input_dims, n_actions,
                       depth, width, activ,
                       noisy,
                       n_atoms ):
        super(C51DuelMLP, self).__init__()

        ## Defining the network features
        self.name       = name
        self.chpt_dir   = chpt_dir
        self.chpt_file  = os.path.join(self.chpt_dir, self.name)
        self.input_dims = input_dims
        self.n_actions  = n_actions

        ## The c51 distributional RL parameters
        self.n_atoms = n_atoms

        # Checking if noisy layers will be used
        if noisy:
            linear_layer = ll.FactNoisyLinear
        else:
            linear_layer = nn.Linear

        ## Defining the base layer structure
        layers = []
        for l_num in range(1, depth+1):
            inpt = input_dims[0] if l_num == 1 else width
            layers.append(( "base_lin_{}".format(l_num), nn.Linear(inpt, width) ))
            layers.append(( "base_act_{}".format(l_num), activ ))
        self.base_stream = nn.Sequential(OrderedDict(layers))

        ## Defining the dueling network arcitecture
        self.V_stream = nn.Sequential(OrderedDict([
            ( "V_lin_1",   linear_layer(width, width) ),
            ( "V_act_1",   activ ),
            ( "V_lin_out", linear_layer(width, n_atoms) ),
        ]))
        self.A_stream = nn.Sequential(OrderedDict([
            ( "A_lin_1",   linear_layer(width, width) ),
            ( "A_act_1",   activ ),
            ( "A_lin_out", linear_layer(width, n_actions*n_atoms) ),
        ]))

        ## Moving the network to the device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, state):
        ## This is a network for the C51 algorithm
        ## So the output is a matrix AxN
            ## Each row is an action
            ## Each column is the prob of a Q atom

        shared_out = self.base_stream(state)
        V = self.V_stream(shared_out).view(-1, 1, self.n_atoms)
        A = self.A_stream(shared_out).view(-1, self.n_actions, self.n_atoms)
        Q = V + A - A.mean( dim=1, keepdim=True)
        Q = F.softmax(Q, dim=-1)

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
                 gamma,       lr,
                 \
                 input_dims,  n_actions,
                 depth, width,
                 activ, noisy,
                 \
                 eps,
                 eps_min,
                 eps_dec ,
                 \
                 mem_size,    batch_size,
                 target_sync, freeze_up,
                 \
                 PER_on,      n_step,
                 PEReps,      PERa,
                 PERbeta,     PERb_inc,
                 PERmax,
                 \
                 n_atoms, sup_range
                 ):

        ## Setting all class variables
        self.__dict__.update(locals())
        self.learn_step_counter = 0


        ## The policy and target networks
        self.policy_net = C51DuelMLP( self.name + "_policy_network", net_dir,
                                      input_dims, n_actions, depth, width, activ,
                                      noisy, n_atoms )
        self.target_net = C51DuelMLP( self.name + "_target_network", net_dir,
                                      input_dims, n_actions, depth, width, activ,
                                      noisy, n_atoms )
        self.target_net.load_state_dict( self.policy_net.state_dict() )

        ## Additional C51 Algorithm variables
        self.vmin = sup_range[0]
        self.vmax = sup_range[1]
        self.delz = (self.vmax-self.vmin) / (self.n_atoms-1)
        self.supports = T.linspace( *sup_range, n_atoms ).to(self.policy_net.device)

        ## The gradient descent algorithm used to train the policy network
        self.optimiser = optim.Adam( self.policy_net.parameters(), lr = lr )

        ## Priotised experience replay for multi-timestep learning
        if PER_on and n_step > 1:
            self.memory = MM.N_Step_PER( mem_size, input_dims,
                                eps=PEReps, a=PERa, beta=PERbeta,
                                beta_inc=PERb_inc, max_priority=PERmax,
                                n_step=n_step, gamma=gamma )

        ## Priotised experience replay
        elif PER_on:
            self.memory = PER( mem_size, input_dims,
                               eps=PEReps, a=PERa, beta=PERbeta,
                               beta_inc=PERb_inc, max_priority=PERmax )

        ## Standard experience replay
        elif n_step == 1:
            self.memory = Experience_Replay( mem_size, input_dims )

        else:
            print( "\n\n!!! Cant do n_step learning without PER !!!\n\n" )
            exit()


    def choose_action(self, state):

        ## Act completly randomly for the first x frames
        if self.memory.mem_cntr < self.freeze_up:
            action = rd.randint(self.n_actions)
            act_dist = np.zeros( self.n_atoms )

        ## If there are no noisy layers then we must do e-greedy
        elif not self.noisy and rd.random() < self.eps:
                action = rd.randint(self.n_actions)
                act_dist = np.zeros( self.n_atoms )
                self.eps = max( self.eps - self.eps_dec, self.eps_min )

        ## Then act purely greedily
        else:
            with T.no_grad():
                state_tensor = T.tensor( [state], device=self.target_net.device, dtype=T.float32 )
                dist = self.policy_net(state_tensor)
                Q_values = T.matmul( dist, self.supports )
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
            pol_Q_next = T.matmul( pol_dist_next, self.supports )

            ## Can then determine the optimum actions
            next_actions = T.argmax(pol_Q_next, dim=1)

            ## We now use the target network to get the distributions of these actions
            tar_dist_next = self.target_net(next_states)[batch_idxes, next_actions]

            ## We can then find the new supports using the distributional Bellman Equation
            new_supports = rewards + ( self.gamma ** self.n_step ) * self.supports * (~dones)
            new_supports = new_supports.clamp(self.vmin, self.vmax)

            ## Must calculate the closest indicies for the projection
            ind = ( new_supports - self.vmin ) / self.delz
            dn = ind.floor().long()
            up = ind.ceil().long()

            ## Also important is where the projections align perfectly
            ones = -T.ones(tar_dist_next.size(), device=self.target_net.device )
            up_is_dn = T.where(up-dn==0, up.float(), ones).long()
            updn_mask = (up_is_dn>-1)
            up_is_dn.clamp_(min=0)

            ## We begin with zeros for the target dist using current supports
            td_target_dist = T.zeros( tar_dist_next.size(), device=self.target_net.device )

            ## We complete the projections using the index_add method and the offsets
            offset = ( T.linspace( 0, (self.batch_size-1)*self.n_atoms, self.batch_size).long()
                                    .unsqueeze(1)
                                    .expand(self.batch_size, self.n_atoms)
                                    .to(self.target_net.device) )

            td_target_dist.view(-1).index_add_( 0, (dn + offset).view(-1),
                                        (tar_dist_next * (up.float() - ind)).view(-1) )
            td_target_dist.view(-1).index_add_( 0, (up + offset).view(-1),
                                        (tar_dist_next * (ind - dn.float())).view(-1) )
            td_target_dist.view(-1).index_add_( 0, (up_is_dn + offset).view(-1),
                                        (tar_dist_next * updn_mask).view(-1) )
            td_target_dist = td_target_dist.detach()

        ## Now we want to track gradients using the policy network
        pol_dist = self.policy_net(states)[batch_idxes, actions]

        ## Calculating the KL Divergence for each sample in the batch
        e = 1e-6
        KLdiv = ( td_target_dist * T.log( e + td_target_dist / (pol_dist+e) ) ).sum(dim=1)

        ## Use the KLDiv as new errors to be used in PER and update the replay
        if self.PER_on:
            new_errors = KLdiv.detach().cpu().numpy().squeeze()
            self.memory.batch_update(indices, new_errors)

        ## Now we use the loss for graidient desc, applying is weights if using PER
        loss = KLdiv
        if self.PER_on:
            loss = loss * is_weights
        loss = loss.mean()
        loss.backward()
        self.optimiser.step()

        self.learn_step_counter += 1

        return loss.item()
