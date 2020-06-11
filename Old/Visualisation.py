import sys
sys.path.append('/home/matthew/Documents/Reinforcement_Learning/')

from RLResources import Layers as ll
from RLResources import MemoryMethods as MM

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict

def gaussian( x, mean=0, sig=1 ):
    return 1/(sig*np.sqrt(2*np.pi)) * np.exp( - (x-mean)**2 / (2*sig**2) )

T.manual_seed(0)

class Categorical_DQN(nn.Module):
    """ A simple and configurable linear duelling dqn model, the stream split is
        done with a single layer using the same width of the network width.
        Unlike my many other networks, this has the optimisor, and the loss fucntion
        build into this class.
    """
    def __init__(self, name, chpt_dir,
                       input_dims, n_outputs,
                       n_atoms, sup_range,
                       learning_rate ):
        super(Categorical_DQN, self).__init__()

        ## Defining the network features
        self.name       = name
        self.chpt_dir   = chpt_dir
        self.chpt_file  = os.path.join(self.chpt_dir, self.name)
        self.input_dims = input_dims
        self.n_outputs  = n_outputs
        self.n_atoms    = n_atoms
        self.supports   = T.linspace( *sup_range, n_atoms )

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
        layers.append(( "lin_out", nn.Linear(width, n_outputs*n_atoms) ))
        self.base_stream = nn.Sequential(OrderedDict(layers))

        self.optimiser = optim.Adam( self.parameters(), lr = learning_rate )
        self.loss_fn = nn.SmoothL1Loss( reduction = "none" )

    def forward(self, state):
        out   = self.base_stream(state).view(-1, self.n_outputs, self.n_atoms)
        probs = F.softmax(out, dim=-1).clamp(min=1e-5)
        return probs

def project( distribution, support, new_support ):
    return 0

batch_size = 4
n_actions = 1
n_atoms = 51
gamma = 0.9
vmin = -10
vmax = 10
sups = T.linspace( vmin, vmax, n_atoms )
delz = (vmax-vmin) / (n_atoms-1)

network = Categorical_DQN( "test", "none",
                            [5], n_actions,
                            n_atoms, [vmin, vmax],
                            1e-5 )

# optimiser = optim.SGD( network.parameters(), lr=1e-1, momentum=0.9, nesterov=False )
# optimiser = optim.ASGD( network.parameters(), lr=1e-1 )
# optimiser = optim.Adadelta( network.parameters() )
# optimiser = optim.Adagrad( network.parameters(), lr=5e-2 )
# optimiser = optim.RMSprop( network.parameters(), lr=5e-3 )
optimiser = optim.Adam( network.parameters(), lr=2e-3 )


plt.ion()
fig = plt.figure( figsize = (5,5) )
ax  = fig.add_subplot(111)
targ_lines = []
pred_lines = []
for i in range(batch_size):
    targ_line, = ax.plot( [], [], "-", color=(i/3,0.5-i/6,1-i/3) )
    pred_line, = ax.plot( [], [], "--", color=(i/3,0.5-i/6,1-i/3) )
    targ_lines.append(targ_line)
    pred_lines.append(pred_line)
print()

## We need to pull the transition from the "memory"
actions     = T.tensor( [0, 0, 0, 0], dtype=T.int64 )
states      = T.rand( (4, 5), dtype=T.float32 )

batch_idxes = list(range(batch_size))

## Changing the Target Distribution To be a Gaussian
target_dist = []
target_dist.append( gaussian(sups, mean=-6) )
target_dist.append( gaussian(sups, mean=-2) )
target_dist.append( gaussian(sups, mean=2) )
target_dist.append( gaussian(sups, mean=6) )
target_dist = T.stack(target_dist)
target_dist /= target_dist.sum(dim=1, keepdim=True)

for i in range(batch_size):
    targ_lines[i].set_data( sups.detach(), target_dist[i].detach() )

ax.relim()
ax.autoscale_view()

## The training loop
loss_fn = nn.SmoothL1Loss()
loss_fn = nn.MSELoss()
loss_fn = nn.L1Loss()
while True:
    fig.canvas.draw()
    fig.canvas.flush_events()

    q_policy = network(states)[batch_idxes, actions]

    optimiser.zero_grad()

    # loss = loss_fn( target_dist, q_policy )
    loss = -(target_dist.detach() * T.log(q_policy)).sum()

    loss.backward()
    optimiser.step()

    for i in range(batch_size):
        pred_lines[i].set_data( sups.detach(), q_policy[i].detach() )

    fig.canvas.draw()
    fig.canvas.flush_events()












