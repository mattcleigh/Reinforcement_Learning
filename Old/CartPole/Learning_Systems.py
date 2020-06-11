import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from collections import namedtuple


Transition = namedtuple("Transition",
                       ("state", "action", "next_state", "reward" ))


class ReplayMemory:
    """ A class for storing, sampling and moving transitions
    """
    def __init__(self, capacity):
        self.capacity  = capacity
        self.memory    = []
        self.position  = 0
        self.long_term = int(capacity/10)

    def push(self, *args):
        """ Saves a given transition to memory in a cyclic manner
        """
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.position = (self.position + 1) % ( self.capacity - self.long_term )
            self.memory[self.position + self.long_term] = Transition(*args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class SimpleNet(nn.Module):
    """ A configurable, but very simple neural network for basic testing
    """
    def __init__(self, activ="ELU", inputs=4, outputs=2, depth=5, width=128, dropout_p=0.5, batch_norm=False ):
        super(SimpleNet, self).__init__()

        self.inputs  = inputs
        self.outputs = outputs

        ####### Defining the activation function #######
        if   activ == "ELU":   activ = nn.ELU()
        elif activ == "ReLU":  activ = nn.ReLU()
        elif activ == "SELU":  activ = nn.SELU()
        elif activ == "GELU":  activ = nn.GELU()
        elif activ == "PReLU": activ = nn.PReLU()
        else:
            print( "\n\nWarning: unrecognised activation function!!!\n\n" )
            return 1

        ####### Defining the normalisation method #######
        if   dropout_p >  0 and not batch_norm:  BN_DP_split = 0
        elif dropout_p >  0 and     batch_norm:  BN_DP_split = math.ceil(depth/2)
        elif dropout_p == 0 and     batch_norm:  BN_DP_split = depth

        ####### Defining the layer structure #######
        layers = []
        for l_num in range(1, depth+1):

            inpt = inputs if l_num == 1 else width

            layers.append(( "lin_{}".format(l_num), nn.Linear(inpt, width) ))
            layers.append(( "act_{}".format(l_num), activ ))

            if dropout_p > 0.0 and l_num > BN_DP_split:  layers.append(( "dro_{}".format(l_num), nn.Dropout(dropout_p) ))
            if batch_norm      and l_num <= BN_DP_split: layers.append(( "btn_{}".format(l_num), nn.BatchNorm1d( width, momentum=0.01 ) ))

        layers.append(( "lin_out", nn.Linear(width, outputs) ))
        layers.append(( "act_out", activ ))
        self.fc = nn.Sequential(OrderedDict(layers))
        ############################################

    def forward(self, data):
        return self.fc(data)



def e_greedy_action( network, state, eps, device ):
    if random.random() < eps:
        return torch.tensor( [[random.randrange(network.outputs)]], device=device, dtype = torch.long )
    else:
        with torch.no_grad():
            return network(state).argmax().view(1,1)


def train( policy_net, target_net, memory, batch_size, gamma, loss_fn, optimizer, last_trans, device ):

    ## We dont train until the memory is at least one batch_size
    if len(memory) < batch_size:
        return

    ## Collect the batch and stich them together elementwise
    transitions     = memory.sample( batch_size )
    transitions[-1] = last_trans
    batch = Transition(*zip(*transitions))


    ## Compute a mask of states that are non-terminal
    has_next_state = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    ## Collect all next states so long as they are not None
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    ## Convert each component of the transition into an tensor
    state_batch  = torch.cat(batch.state).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    action_batch = torch.cat(batch.action).to(device)

    ## We compute the estimates of the policy network
    ## Remeber the output is the q values of both actions
    ## So we use gather to get the estimate of the action that was chosen
    state_q_values = policy_net(state_batch).gather(1, action_batch)

    ## Now we compute the target q values
    next_q_values = torch.zeros((batch_size, 1), device=device).to(device)
    next_q_values[has_next_state] = target_net(non_final_next_states).max(1)[0].detach().unsqueeze(1)
    target_q_values = reward_batch + gamma * next_q_values


    ## Now we calculate the loss
    loss = loss_fn( state_q_values, target_q_values ).mean()
    # print(loss)

    ## Now we optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad.clip_grad_norm_(policy_net.parameters(), 10)
    # for param in policy_net.parameters():
        # param.grad.data.clamp_(-1, 1)
    optimizer.step()

























