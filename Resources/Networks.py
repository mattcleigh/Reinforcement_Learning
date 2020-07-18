import os
import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class TwinCriticMLP(nn.Module):
    """ A couple of simple and configurable multilayer perceptrons.
        One class contains both critic networks used in TD3
    """
    def __init__(self, name, chpt_dir,
                       input_dims, n_actions,
                       depth, width, activ ):
        super(TwinCriticMLP, self).__init__()

        ## Defining the network features
        self.name       = name
        self.chpt_dir   = chpt_dir
        self.chpt_file  = os.path.join(self.chpt_dir, self.name)
        self.input_dims = input_dims
        self.n_actions  = n_actions

        ## The layer structure of the first critic
        layers = []
        for l_num in range(1, depth+1):
            inpt = (n_actions+input_dims[0]) if l_num == 1 else width
            layers.append(( "crit_1_lin_{}".format(l_num), nn.Linear(inpt, width) ))
            layers.append(( "crit_1_act_{}".format(l_num), activ ))
        layers.append(( "crit_1_lin_out", nn.Linear(width, 1) ))
        self.crit_layers_1 = nn.Sequential(OrderedDict(layers))

        ## The layer structure of the second critic
        layers = []
        for l_num in range(1, depth+1):
            inpt = (n_actions+input_dims[0]) if l_num == 1 else width
            layers.append(( "crit_2_lin_{}".format(l_num), nn.Linear(inpt, width) ))
            layers.append(( "crit_2_act_{}".format(l_num), activ ))
        layers.append(( "crit_2_lin_out", nn.Linear(width, 1) ))
        self.crit_layers_2 = nn.Sequential(OrderedDict(layers))

        ## Moving the network to the device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        state_action = T.cat((state, action), 1 )
        q1 = self.crit_layers_1(state_action)
        q2 = self.crit_layers_2(state_action)
        return q1, q2

    def Q1_only(self, state, action):
        state_action = T.cat((state, action), 1 )
        q1 = self.crit_layers_1(state_action)
        return q1

    def save_checkpoint(self, flag=""):
        print("... saving critic network checkpoint ..." )
        T.save(self.state_dict(), self.chpt_file+flag)

    def load_checkpoint(self, flag=""):
        print("... loading critic network checkpoint ..." )
        self.load_state_dict(T.load(self.chpt_file+flag))

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

class FactNoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_zero=0.5, bias=True):
        super(FactNoisyLinear, self).__init__(in_features, out_features, bias=bias)

        ## Calculate the hyperparameter for the noisy parameter initialisation
        sigma_init = sigma_zero / math.sqrt(in_features)

        ## Create the noisy weight parameters
        self.weight_noisy = nn.Parameter( T.Tensor(out_features, in_features).fill_(sigma_init) )

        ## We do the same for the bias if there is one in the layer
        if bias:
            self.bias_noisy = nn.Parameter( T.Tensor(out_features).fill_(sigma_init) )

        ## The random numbers are not trained by the optimiser, so must be registered as buffers
        self.register_buffer("eps_input",  T.zeros( 1,  in_features ) )
        self.register_buffer("eps_output", T.zeros( out_features, 1 ) )

    def forward(self, input):

        ## For each forward pass we must generate new random numbers
        T.randn( self.eps_input.size(),  out=self.eps_input  )
        T.randn( self.eps_output.size(), out=self.eps_output )

        ## In Factorised Noisy Layers the random numbers are modified by a small function
        func = lambda x: T.sign(x) * T.sqrt(T.abs(x))
        f_eps_in  = func( self.eps_input  )
        f_eps_out = func( self.eps_output )

        ## Now we can apply the random numbers the layer information
        full_weight = self.weight + self.weight_noisy * T.mul(f_eps_in,f_eps_out)
        full_bias   = self.bias
        if full_bias is not None:
            full_bias = self.bias + self.bias_noisy * f_eps_out.t().squeeze()

        return F.linear( input, full_weight, full_bias )
