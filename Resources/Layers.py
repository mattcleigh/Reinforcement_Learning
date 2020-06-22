import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F

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
