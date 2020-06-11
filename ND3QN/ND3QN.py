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

class ND_DQN(nn.Module):
    """ A simple and configurable linear duelling dqn model, the stream split is
        done with a single layer using the same width of the network width.
        The last layers are noisy.
    """
    def __init__(self, name, chpt_dir,
                       input_dims, n_outputs ):
        super(ND_DQN, self).__init__()

        ## Defining the network features
        self.name       = name
        self.chpt_dir   = chpt_dir
        self.chpt_file  = os.path.join(self.chpt_dir, self.name)
        self.input_dims = input_dims
        self.n_outputs  = n_outputs

        ## Network Parameters (should probably make these arguments to the agent as well
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
            ( "V_lin_out", ll.FactNoisyLinear(width//2, 1) ),
        ]))
        self.A_stream = nn.Sequential(OrderedDict([
            ( "A_lin_1",   ll.FactNoisyLinear(width, width//2) ),
            ( "A_act_1",   activ ),
            ( "A_lin_out", ll.FactNoisyLinear(width//2, n_outputs) ),
        ]))

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
        self.input_dims  = input_dims
        self.n_actions   = n_actions
        self.net_dir     = net_dir

        ## The policy and target Noisy+Duelling+DQN networks
        self.policy_net = ND_DQN( self.name + "_policy_network", net_dir,
                                        input_dims, n_actions )

        self.target_net = ND_DQN( self.name + "_target_network", net_dir,
                                        input_dims, n_actions )

        ## The gradient descent algorithm and loss function used to train the policy network
        self.optimiser = optim.Adam( self.policy_net.parameters(), lr = lr )
        self.loss_fn = nn.SmoothL1Loss( reduction = "none" )

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
            act_value = 0

        ## Then act purely greedily, exploration comes due to noisy layers
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

        ## Remember that the output is the Q values of all actions
        ## So we use "gather" to get the estimate of the action that was chosen

        ## To increase the speed of this step we do it without keeping track of gradients
        with T.no_grad():

            ## For double Q-learning we calculate the best action for the next state using the policy network
            ## But we get the value of the next state from the target network
            next_actions  = T.argmax( self.policy_net(next_states), dim=1 )
            Q_next        = self.target_net(next_states).gather( 1, next_actions.unsqueeze(1) )
            Q_next[dones] = 0.0

            ## Calculate the target values based on the bellman equation
            td_targets = rewards.unsqueeze(1) + ( self.gamma ** 3 ) * Q_next.detach()

        ## Now we calculate the network estimates for the state values
        Q_states = self.policy_net(states).gather( 1, actions.unsqueeze(1) )

        ## Calculate the TD-Errors to be used in PER and update the replay
        new_errors = T.abs(Q_states - td_targets).detach().cpu().numpy().squeeze()
        self.memory.batch_update(indices, new_errors)
        error = new_errors.mean()

        ## Calculate the loss individually for each element and perform graidient desc
        loss = self.loss_fn( Q_states, td_targets )
        loss = loss * is_weights.unsqueeze(1)
        loss = loss.mean()
        loss.backward()
        self.optimiser.step()

        self.learn_step_counter += 1

        return loss.item(), error






















