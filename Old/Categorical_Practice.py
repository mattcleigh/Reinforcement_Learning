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
        depth = 1
        width = 5
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
n_atoms = 11
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
optimiser = optim.Adam( network.parameters(), lr=5e-2 )


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
rewards     = T.tensor( [0, 1, 2, 4], dtype=T.int64 )
dones       = T.tensor( [ 0, 0, 1, 1 ], dtype=T.bool )
states      = T.rand( (4, 5), dtype=T.float32 )
next_states = T.tensor( [ [ 1, 2, 3, 4, 5 ],
                         [ 2, 4, 6, 7, 8 ],
                         [ 5, 4, 3, 2, 1 ],
                         [ 1, 3, 5, 7, 9 ] ], dtype=T.float32 )

# next_state = T.tensor( [ 1, 2, 3, 4, 5 ], dtype=T.float32 )
# next_state = T.tensor( [ 5, 4, 3, 2, 1 ], dtype=T.float32 )

## Feed the next state into the network to get the target distribution
next_matrix = network(next_states)
print("Next Matrix")
print(next_matrix.detach().numpy())
print()

## Compute the q values using the support
q_next = T.matmul( next_matrix, sups )
print("Next Q-Values")
print(q_next.detach().numpy())
print()

## Determine the optimum action
next_actions = T.argmax(q_next, dim=1)
print("Next Actions")
print(next_actions.detach().numpy())
print()

## Get the distribution corresponding to the next action
batch_idxes = list(range(batch_size))
next_dist = next_matrix[batch_idxes, next_actions]
print("Next Q-Probabilities")
print(next_dist.detach().numpy())
print()

## We calculate the supports of the target dist
print("Old Supports")
print(sups.detach().numpy())

rewards = rewards.reshape(-1, 1)
dones = dones.reshape(-1, 1)
new_supports = rewards + gamma * sups * (~dones)
new_supports = new_supports.clamp(vmin, vmax)
print("New Supports")
print(new_supports.detach().numpy())
print()

## We must calculate the closest indicies for the projection
ind = ( new_supports - vmin ) / delz
dn = ind.floor().long()
up = ind.ceil().long()
up_is_dn = T.where(up-dn==0, up.float(), -T.ones(next_dist.size()) ).long()
updn_mask = (up_is_dn>-1)
up_is_dn.clamp_(min=0)

## We begin with zeros for the target dist using current supports
target_dist = T.zeros( next_dist.size() )

offset = ( T.linspace( 0, (batch_size-1)*n_atoms, batch_size).long()
            .unsqueeze(1)
            .expand(batch_size, n_atoms) )

target_dist.view(-1).index_add_( 0, (dn + offset).view(-1),
                            (next_dist * (up.float() - ind)).view(-1) )
target_dist.view(-1).index_add_( 0, (up + offset).view(-1),
                            (next_dist * (ind - dn.float())).view(-1) )
target_dist.view(-1).index_add_( 0, (up_is_dn + offset).view(-1),
                            (next_dist * updn_mask).view(-1) )

print("Target Distribution")
print(target_dist.detach().numpy())
print()

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
# loss_fn = nn.SmoothL1Loss()
# loss_fn = nn.MSELoss()
# loss_fn = nn.L1Loss()
# while True:
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#
#     q_policy = network(states)[batch_idxes, actions]
#     # print("Policy Distribution")
#     # print(q_policy.detach().numpy())
#     # print()
#
#     optimiser.zero_grad()
#     loss = -(target_dist.detach() * T.log(q_policy)).sum(dim=1)
#     print(loss)
#     # loss = loss_fn( target_dist, q_policy )
#     loss=loss.sum()
#     loss.backward()
#     optimiser.step()
#
#     for i in range(batch_size):
#         pred_lines[i].set_data( sups.detach(), q_policy[i].detach() )
#
#     fig.canvas.draw()
#     fig.canvas.flush_events()












