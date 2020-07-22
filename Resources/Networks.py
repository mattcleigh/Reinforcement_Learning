import os
import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class TwinCriticMLP(nn.Module):
    """ A single class that contains both critic networks used in TD3 and SAC.
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

        ## The layer structure of the two critics
        tot_inpt = n_actions + input_dims[0]
        self.crit_layers_1 = mlp_creator( "crit_1", n_in=tot_inpt, n_out=1, d=depth, w=width, act_h=activ )
        self.crit_layers_2 = mlp_creator( "crit_2", n_in=tot_inpt, n_out=1, d=depth, w=width, act_h=activ )

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
    """ A network system used in A2C and PPO.
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

        ## Defining the base (shared) layer structure,
        ## the actor network, returning the policy using a softmax function
        ## and the critic network, which returns the state value function V
        self.base_stream = mlp_creator( "base", n_in=input_dims[0], w=width, d=depth, act_h=activ )
        self.actor_stream = mlp_creator( "actor", n_in=width, n_out=n_actions, w=width, act_h=activ, act_o=nn.Softmax(dim=-1) )
        self.critic_stream = mlp_creator( "critic", n_in=width, n_out=1, w=width, act_h=activ )

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


def mlp_creator( name, n_in=1, n_out=None, d=1, w=256,
                       act_h=nn.ReLU(), act_o=None, nsy=False, l_nrm=False,
                       custom_size=None, return_list=False ):
    """ A function used by many of the project algorithms to contruct a
        simple and configurable MLP.

        By default the function returns the full nn sequential model, but if
        return_list is set to true then the output will still be in list form
        to allow final layer configuration by the caller.

        The custom_size argument is a list for creating streams with varying
        layer width. If this is set then the width and depth parameters
        will be ignored.
    """
    layers = []
    widths = []

    ## Generating an array to use as the widths
    widths.append( n_in )
    if custom_size is not None:
        d = len( custom_size )
        widths += custom_size
    else:
        widths += d*[w]

    # Checking if noisy layers will be used
    if nsy:
        linear_layer = FactNoisyLinear
    else:
        linear_layer = nn.Linear

    ## Creating the "hidden" layers in the stream
    for l in range(1, d+1):
        layers.append(( "{}_lin_{}".format(name, l), linear_layer(widths[l-1], widths[l]) ))
        layers.append(( "{}_act_{}".format(name, l), act_h ))
        if l_nrm:
            layers.append(( "{}_nrm_{}".format(name, l), nn.LayerNorm(widths[l]) ))

    ## Creating the "output" layer of the stream if applicable which is sometimes
    ## Not the case when creating base streams in larger arcitectures
    if n_out is not None:
        layers.append(( "{}_lin_out".format(name), linear_layer(widths[-1], n_out) ))
        if act_o is not None:
            layers.append(( "{}_act_out".format(name), act_o ))

    ## Return the list of features or...
    if return_list:
        return layers

    ## ... convert the list to an nn, then return
    return nn.Sequential(OrderedDict(layers))
