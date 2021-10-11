import math
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Resources import utils as myUT

class NoisyLinear(nn.Linear):
    ''' A pytorch linear layer with factorised gaussian noise injection
    The layer contains learnable parameters to dictate how much it listens to the noise injection
    '''
    def __init__(self, n_inp, n_out, sigma_zero=0.5, bias=True):
        ''' Constructor method for NoisyLinear
        args:
            n_inp: Size of input vector
            n_out: Size of output vector
        kwargs:
            sigma_zero: The initial deviation of the noise
            bias: If the linear layer includes a bias parameter
        '''
        ## Class inherits from the standard pytorch linear layer
        super().__init__(n_inp, n_out, bias=bias)

        ## The deviation of the noise is scaled by the number of inputs
        sigma_init = sigma_zero / math.sqrt(n_inp)

        ## The parameters which contribute noise to the linear weights
        self.weight_noisy = nn.Parameter(T.full((n_out, n_inp), sigma_init))

        ## The parameters which contribute noise to the linear biases
        if bias:
            self.bias_noisy = nn.Parameter(T.full((n_out,), sigma_init))

        ## The random numbers themselves are not trained by the optimiser, so must be registered as buffers
        self.register_buffer("eps_input",  T.zeros(1, n_inp))
        self.register_buffer("eps_output", T.zeros(n_out, 1))

    def forward(self, input):

        ## For each forward pass we must generate new random numbers
        T.randn( self.eps_input.size(),  out=self.eps_input  )
        T.randn( self.eps_output.size(), out=self.eps_output )

        ## In "Factorised Noisy Layers" the random numbers are modified by a small function to minimise space
        func = lambda x: T.sign(x) * T.sqrt(T.abs(x))
        f_eps_in  = func( self.eps_input  )
        f_eps_out = func( self.eps_output )

        ## Now we can apply the random numbers the layer information
        full_weight = self.weight + self.weight_noisy * T.mul(f_eps_in, f_eps_out)
        full_bias   = self.bias
        if full_bias is not None:
            full_bias = self.bias + self.bias_noisy * f_eps_out.t().squeeze()

        return F.linear(input, full_weight, full_bias)

class MLPBlock(nn.Module):
    ''' A single block used in a dense network or an mlp
    Produces a nn.sequential that applies the following transformations:
        - linear layer (may be noise)
        - actionvation
        - layer normalisaion
        - dropout
    '''
    def __init__(self, n_in, n_out, act, nrm, drp, nsy):
        super().__init__()

        ## Which type of layer will be used
        lyr = NoisyLinear if nsy else nn.Linear

        ## Build the block
        block = [ lyr(n_in, n_out) ]
        if act: block += [ myUT.get_act(act) ]
        if nrm: block += [ nn.LayerNorm(n_out) ]
        if drp: block += [ nn.Dropout(drp) ]
        self.block = nn.Sequential(*block)

    def forward(self, data):
        return self.block(data)

def mlp_blocks( n_in=1, n_out=1, depth=2, width=32,
                act_h='lrlu', act_o=None,
                nrm=False, drp=0, nsy=False,
                widths=[], as_list=False ):
    ''' Returns a series of consecutive MLP blocks within a nn.Squential module
    kwargs:
        n_in:   The number of input features for the mlp
        n_out:  The number of output features for the mlp
        depth:  The number of hidden MLP blocks
        width:  The width of the hidden MLP blocks
        act_h:  The activation function in the hidden layers
        act_o:  The activation function of the output layer
        nrm:    If layer normalisation will be applied in the hidden layers
        drp:    If dropout will be applied in the hidden layers
        nsy:    If the linear layers will be noisy
        widths: A list of widths, overrides the width and depth argument
        as_list: If the blocks should be returned as a list
    returns:
        The MLP blocks as a nn.sequential object or as a list depending on the as_list argument
    '''

    ## The widths argument overrides width and depth
    if not widths:
        widths = depth * [ width ]

    ## Input block
    blocks = [ MLPBlock(n_in, widths[0], act_h, nrm, drp, nsy) ]

    ## Hidden blocks
    for w1, w2 in zip(widths[:-1], widths[1:]):
        blocks += [ MLPBlock(w1, w2, act_h, nrm, drp, nsy) ]

    ## Output block
    if n_out:
        blocks += [ MLPBlock(widths[-1], n_out, act_o, False, 0, nsy) ]

    ## Return as a list or a pytorch sequential object
    if as_list:
        return blocks
    else:
        return nn.Sequential(*blocks)
