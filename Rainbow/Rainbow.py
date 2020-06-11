import sys
sys.path.append('/home/matthew/Documents/Reinforcement_Learning/')

from RLResources import Layers as ll
from RLResources import MemoryMethods as MM

import os
import time
import numpy as np

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict

T.manual_seed(0)

class RainbowNet(nn.Module):
    """ A simple and configurable linear DQN for Rainbow learning.
        A rainbow network = distributional + duelling + noisy
        This network contains seperate streams for value and advantage evaluation
        using noisy linear layers.
        It is also distributional, so it produces probability estimates over a
        support rather than the Q values themselves.
    """
    def __init__(self, name, chpt_dir,
                       input_dims, n_outputs,
                       n_atoms, sup_range, ):
        super(RainbowNet, self).__init__()

        ## Defining the network features
        self.name       = name
        self.chpt_dir   = chpt_dir
        self.chpt_file  = os.path.join(self.chpt_dir, self.name)
        self.input_dims = input_dims
        self.n_outputs  = n_outputs

        ## The distributional RL parameters
        self.n_atoms = n_atoms
        self.supports = T.linspace( *sup_range, n_atoms )

        ## Network structure rarameters (should probably make these arguments)
        depth = 2
        width = 128
        activ = nn.PReLU()

        ## Defining the shared layer structure
        layers = []
        for l_num in range(1, depth+1):
            inpt = input_dims[0] if l_num == 1 else width
            layers.append(( "lin_{}".format(l_num), nn.Linear(inpt, width) ))
            layers.append(( "act_{}".format(l_num), activ ))
        self.base_stream = nn.Sequential(OrderedDict(layers))

        ## Defining the duelling network arcitecture
        self.V_stream = nn.Sequential(OrderedDict([
            ( "V_lin_1",   ll.FactNoisyLinear(width, width//2) ),
            ( "V_act_1",   activ ),
            ( "V_lin_out", ll.FactNoisyLinear(width//2, n_atoms) ),
        ]))
        self.A_stream = nn.Sequential(OrderedDict([
            ( "A_lin_1",   ll.FactNoisyLinear(width, width//2) ),
            ( "A_act_1",   activ ),
            ( "A_lin_out", ll.FactNoisyLinear(width//2, n_outputs*n_atoms) ),
        ]))

        ## Moving the network to the device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        self.supports = self.supports.to(self.device)


    def forward(self, state):
        ## Remember that the output is matrix AxN, where each row represents an action and
        ## each column represents the probability of an atom representing a Q value

        shared_out = self.base_stream(state)
        V = self.V_stream(shared_out).view(-1, 1, self.n_atoms)
        A = self.A_stream(shared_out).view(-1, self.n_outputs, self.n_atoms)
        Q = V + A - A.mean( dim=1, keepdim=True)

        Q = F.softmax(Q, dim=-1)
        Q = Q.clamp(min=1e-3) # For avoiding NaNs

        return Q

    def save_checkpoint(self, flag=""):
        print("... saving network checkpoint ..." )
        T.save(self.state_dict(), self.chpt_file+flag)

    def load_checkpoint(self, flag=""):
        print("... loading network checkpoint ..." )
        self.load_state_dict(T.load(self.chpt_file+flag))


class Agent(object):
    """ The agent is the object that navigates the envirnoment, it is equipped with (two)
        DQN(s) to gather values, but it is not the DQN itself
    """
    def __init__(self, name,
                       gamma,       lr,
                       input_dims,  n_actions,
                       mem_size,    batch_size,
                       target_sync, freeze_up,
                       PEReps,      PERa,
                       PERbeta,     PERb_inc,
                       PERmax_td,   n_step,
                       n_atoms,     sup_range,
                       net_dir = "tmp/dqn"):

        ## General learning attributes
        self.name        = name
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.target_sync = target_sync
        self.freeze_up   = freeze_up
        self.n_step      = n_step
        self.learn_step_counter = 0

        ## DQN parameters
        self.net_dir     = net_dir
        self.input_dims  = input_dims
        self.n_actions   = n_actions

        ## Categorical/Distributional DQN parameters
        self.n_atoms  = n_atoms
        self.vmin     = sup_range[0]
        self.vmax     = sup_range[1]
        self.delz     = (self.vmax-self.vmin) / (self.n_atoms-1)

        ## The policy and target Dist+Noisy+Duelling=Rainbow networks
        self.policy_net = RainbowNet( self.name + "_policy_network", net_dir,
                            input_dims, n_actions,
                            n_atoms, sup_range )

        self.target_net = RainbowNet( self.name + "_target_network", net_dir,
                            input_dims, n_actions,
                            n_atoms, sup_range )

        ## The gradient descent algorithm used to train the policy network
        self.optimiser = optim.Adam( self.policy_net.parameters(), lr = lr )

        ## Priotised experience replay for multi-timestep learning
        self.memory = MM.N_Step_PER( mem_size, input_dims,
                            eps=PEReps, a=PERa, beta=PERbeta,
                            beta_inc=PERb_inc, max_tderr=PERmax_td,
                            n_step=n_step, gamma=gamma )

        ## Make the networks synced before commencing training
        self.target_net.load_state_dict( self.policy_net.state_dict() )

    def choose_action(self, state):

        ## Act completly randomly for the first x frames
        if self.memory.mem_cntr < self.freeze_up:
            action = np.random.randint(self.n_actions)
            act_dist = np.zeros( self.n_atoms )

        ## Then act purely greedily, exploration comes due to noisy layers
        else:
            with T.no_grad():
                state_tensor = T.tensor( [state], device=self.target_net.device, dtype=T.float32 )
                dist = self.policy_net(state_tensor)
                Q_values = T.matmul( dist, self.policy_net.supports )
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
            return 0, 0

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
        ## Since the gather method wont work with these higher dimension outputs
        batch_idxes = list(range(self.batch_size))

        ## To increase the speed of this step we do it without keeping track of gradients
        with T.no_grad():

            ## First we need the next state distribution using the policy network for double Q learning
            pol_dist_next = self.policy_net(next_states)

            ## We then find the Q-values of the actions by summing over the supports
            pol_Q_next = T.matmul( pol_dist_next, self.policy_net.supports )

            ## Can then determine the optimum actions
            next_actions = T.argmax(pol_Q_next, dim=1)

            ## We now use the target network to get the distributions of these actions
            tar_dist_next = self.target_net(next_states)[batch_idxes, next_actions]

            ## We can then find the new supports using the distributional Bellman Equation
            rewards = rewards.reshape(-1, 1)
            dones = dones.reshape(-1, 1)
            new_supports = rewards + ( self.gamma ** self.n_step ) * self.policy_net.supports * (~dones)
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
            target_dist = T.zeros( tar_dist_next.size(), device=self.target_net.device )

            ## We complete the projections using the index_add method and the offsets
            offset = ( T.linspace( 0, (self.batch_size-1)*self.n_atoms, self.batch_size).long()
                                    .unsqueeze(1)
                                    .expand(self.batch_size, self.n_atoms)
                                    .to(self.target_net.device) )

            target_dist.view(-1).index_add_( 0, (dn + offset).view(-1),
                                        (tar_dist_next * (up.float() - ind)).view(-1) )
            target_dist.view(-1).index_add_( 0, (up + offset).view(-1),
                                        (tar_dist_next * (ind - dn.float())).view(-1) )
            target_dist.view(-1).index_add_( 0, (up_is_dn + offset).view(-1),
                                        (tar_dist_next * updn_mask).view(-1) )
            target_dist = target_dist.detach()

        ## Now we want to track gradients using the policy network
        pol_dist = self.policy_net(states)[batch_idxes, actions]

        ## Calculating the KL Divergence for each sample in the batch
        KLdiv = -(target_dist * T.log(pol_dist)).sum(dim=1)

        ## Use the KLDiv as new errors to be used in PER and update the replay
        new_errors = KLdiv.detach().cpu().numpy().squeeze()
        self.memory.batch_update(indices, new_errors)
        error = new_errors.mean()

        ## Now we use the KLDiv for gradient descent
        loss = KLdiv * is_weights
        loss = loss.mean()
        loss.backward()
        self.optimiser.step()

        self.learn_step_counter += 1

        return loss.item(), error






















