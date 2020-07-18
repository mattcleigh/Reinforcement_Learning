import sys
home_env = '../../../../Reinforcement_Learning/'
sys.path.append(home_env)

from Resources import Networks as myNN
from Resources import MemoryMethods as myMM
from Resources import Utils as myUT

import os
import time
import numpy as np
import numpy.random as rd
from collections import OrderedDict

import torch as T
import torch.nn as nn
import torch.optim as optim

class ActorNetwork(nn.Module):
    """ A simple and configurable multilayer perceptron.
        Tanh applied on final layer to clip the output.
        Scaling can then happen in post depending on env.
    """
    def __init__(self, name, chpt_dir,
                       input_dims, n_actions,
                       depth, width, activ):
        super(ActorNetwork, self).__init__()

        ## Defining the network features
        self.name       = name
        self.chpt_dir   = chpt_dir
        self.chpt_file  = os.path.join(self.chpt_dir, self.name)
        self.input_dims = input_dims
        self.n_actions  = n_actions

        layers = []
        for l_num in range(1, depth+1):
            inpt = input_dims[0] if l_num == 1 else width
            layers.append(( "lin_{}".format(l_num), nn.Linear(inpt, width) ))
            layers.append(( "act_{}".format(l_num), activ ))
        layers.append(( "lin_out", nn.Linear(width, 2*n_actions) ))
        self.main_stream = nn.Sequential(OrderedDict(layers))

        ## Moving the network to the device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, get_ent=True):

        ## The state is pushed through the network and returns [mean1, mean2..., logstd1, logstd2, ... ]
        output = self.main_stream(state)

        ## The output is split into the seperate means and stds components
        means, log_stds = T.chunk(output, 2, dim=-1)

        ## The model predicts log_stds to ensure that the std's are positive
        stds = log_stds.exp()

        ## We now create sample the distribution based on these statistics
        gaussian_dist = T.distributions.Normal(means, stds)
        z = gaussian_dist.rsample()

        ## Apply the squashing function to ge the chosen action
        action = T.tanh( z )

        ## If thats all we want then we return
        if not get_ent:
            return action

        ## We continue if we want to calculate the entropies of the chosen samples
        log_probs = gaussian_dist.log_prob( z ) - T.log( 1 - action*action + 1e-6 )
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return action, entropies

    def save_checkpoint(self, flag=""):
        print("... saving actor network checkpoint ..." )
        T.save(self.state_dict(), self.chpt_file+flag)

    def load_checkpoint(self, flag=""):
        print("... loading actor network checkpoint ..." )
        self.load_state_dict(T.load(self.chpt_file+flag))


class Agent(object):
    def __init__(self,
                 name,
                 net_dir,
                 \
                 gamma, ent_coef, ent_tune,
                 input_dims, n_actions,
                 active, grad_clip,
                 \
                 C_lr, C_depth, C_width,
                 A_lr, A_depth, A_width,
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
        self.eps = ent_coef

        ## The twin critics and their corresponding target networks
        self.critic = myNN.TwinCriticMLP( self.name + "_critic", net_dir,
                                       input_dims, n_actions,
                                       C_depth, C_width, active )
        self.t_critic = myNN.TwinCriticMLP( self.name + "_targ_critic", net_dir,
                                         input_dims, n_actions,
                                         C_depth, C_width, active )
        self.t_critic.load_state_dict( self.critic.state_dict() )

        ## The actor. SAC does not use a target actor network
        self.actor = ActorNetwork( self.name + "_actor", net_dir,
                                   input_dims, n_actions,
                                   A_depth, A_width, active )

        ## The gradient descent algorithms and loss function
        self.C_optimiser = optim.Adam( self.critic.parameters(), lr = C_lr )
        self.A_optimiser = optim.Adam( self.actor.parameters(), lr = A_lr )
        self.loss_fn = nn.SmoothL1Loss( reduction = 'none' )

        ## If we are using the adjustable temperature configuration
        if self.ent_tune:
            ## The target entropy is set to -|A|
            self.target_ent = - float( n_actions )
            self.log_ent_coef = T.zeros(1, requires_grad=True, device=self.actor.device )
            self.ent_coef = self.log_ent_coef.exp()
            self.ent_optim = optim.Adam( [self.log_ent_coef], lr=A_lr )

        ## The agent memory
        self.memory = myUT.cont_memory_creator( PER_on, n_step, gamma, mem_size, n_actions,
                                                input_dims, PEReps, PERa,
                                                PERbeta, PERb_inc, PERmax )

    def save_models(self, flag=""):
        self.critic.save_checkpoint(flag)
        self.t_critic.save_checkpoint(flag)
        self.actor.save_checkpoint(flag)

    def load_models(self, flag=""):
        self.critic.load_checkpoint(flag)
        self.t_critic.load_checkpoint(flag)
        self.actor.load_checkpoint(flag)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sync_target_networks(self):
        if self.target_sync>1:
            print("\n\n\nWarning: SAC only supports soft network updates\n\n\n")
            exit()
        for tp, pp in zip( self.t_critic.parameters(), self.critic.parameters() ):
            tp.data.copy_( self.target_sync * pp.data + ( 1.0 - self.target_sync ) * tp.data )

    def choose_action(self, state):

        ## Act completly randomly for the first x frames
        if self.memory.mem_cntr < self.freeze_up:
            action = np.random.uniform( -1, 1, self.n_actions )

        ## Then act purely greedily with a stochastic policy
        else:
            with T.no_grad():
                state_tensor = T.tensor( [state], device=self.actor.device, dtype=T.float32 )
                action = self.actor(state_tensor, get_ent=False).cpu().numpy().squeeze()

        return action, 0

    def train(self):

        ## We dont train until the memory is at least one batch_size
        if self.memory.mem_cntr < max(self.batch_size, self.freeze_up):
            return 0

        ## Collect the batch
        states, actions, rewards, next_states, dones, is_weights, indices = self.memory.sample_memory(self.batch_size)

        ## We need to convert all of these arrays to pytorch tensors
        states      = T.tensor( states,      device = self.actor.device )
        actions     = T.tensor( actions,     device = self.actor.device )
        rewards     = T.tensor( rewards,     device = self.actor.device ).reshape(-1, 1)
        next_states = T.tensor( next_states, device = self.actor.device )
        dones       = T.tensor( dones,       device = self.actor.device ).reshape(-1, 1)
        is_weights  = T.tensor( is_weights,  device = self.actor.device ).reshape(-1, 1)

        ## To increase the speed of this step we do it without keeping track of gradients
        with T.no_grad():

            ## First we need the optimising actions and their entropies using our actor
            next_actions, next_entropies = self.actor(next_states)

            ## Now we find the values of those actions using both target critics and take the min
            ## We also add entropy regularisation
            next_Q_1, next_Q_2 = self.t_critic( next_states, next_actions )
            next_Q_values = T.min(next_Q_1, next_Q_2) + self.ent_coef * next_entropies

            ## Now we can compute the entropy regularised TD targets
            td_target = rewards + self.n_gamma * next_Q_values * (~dones)
            td_target = td_target.detach()

        ## We compute the current Q value estimates using the two critics
        Q_1, Q_2 = self.critic(states, actions)

        ## Update the Q-Function of the twin critic using gradient descent
        self.C_optimiser.zero_grad()
        C_loss = self.loss_fn( Q_1, td_target ) + self.loss_fn( Q_2, td_target )
        C_loss = ( C_loss * is_weights ).mean()
        C_loss.backward()
        self.C_optimiser.step()

        ## Update the policy by one step of gradient ascent
        self.A_optimiser.zero_grad()
        new_actions, new_entropies = self.actor(states)
        new_Q_1, new_Q_2 = self.critic( states, new_actions )
        new_Q_values = T.min(new_Q_1, new_Q_2) + self.ent_coef * new_entropies
        A_loss = - ( new_Q_values * is_weights ).mean()
        A_loss.backward()
        self.A_optimiser.step()

        ## Update the target network parameters
        self.sync_target_networks()

        ## Update the entropy influcence if we are using dynalic temps
        if self.ent_tune:
            ## We want to increase alpha when the entropy is less than target
            self.ent_optim.zero_grad()
            ent_loss = self.log_ent_coef * ( new_entropies - self.target_ent ).detach()
            ent_loss = ( ent_loss * is_weights ).mean()
            ent_loss.backward()
            self.ent_optim.step()
            self.ent_coef = self.log_ent_coef.exp()
            self.eps = self.ent_coef.item()

        ## Calculate the TD-Errors using both networks to be used in PER and update the replay
        if self.PER_on:
            err_1 = T.abs(Q_1 - td_target)
            err_2 = T.abs(Q_2 - td_target)
            max_err = T.max(err_1, err_2).detach().cpu().numpy().squeeze()
            self.memory.batch_update(indices, max_err)

        self.learn_step_counter += 1

        return C_loss
