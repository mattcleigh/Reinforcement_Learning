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

    def push(self, *args):
        """ Saves a given transition to memory in a cyclic manner
        """
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.position] = Transition(*args)
            self.position = (self.position + 1) % ( self.capacity )

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class SimpleNet(nn.Module):
    """ A configurable, but very simple neural network for basic testing
    """
    def __init__(self, activ="PReLU", inputs=23, outputs=6, depth=2, width=64, dropout_p=0.0, batch_norm=False ):
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


def train( policy_net, target_net, memory, batch_size, gamma, loss_fn, optimiser, last_trans, device ):

    ## We dont train until the memory is at least one batch_size
    if len(memory) < batch_size:
        return

    ## Collect the batch and stich them together elementwise, making sure the most recent one is present in the batch
    transitions     = memory.sample( batch_size )
    transitions[-1] = last_trans
    batch = Transition(*zip(*transitions))

    ## Compute a mask of states that are non-terminal
    has_next_state = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)

    ## Collect all next states so long as the next state is not none
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    ## Convert each component of the transition into an tensor
    state_batch  = torch.cat(batch.state).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    action_batch = torch.cat(batch.action).to(device)

    ## We compute the estimates of the policy network
    ## Remeber the output is the q values of all actions
    ## So we use gather to get the estimate of the action that was chosen
    state_q_values = policy_net(state_batch).gather(1, action_batch)

    ## For double Q-learning we calculate the best action for the next state using the policy network
    next_actions  = torch.t( torch.argmax( policy_net(non_final_next_states), 1 ).unsqueeze(0) )

    ## But then we use the target network to chose those values
    next_action_values = target_net(non_final_next_states).gather(1, next_actions)

    ## We then calculate the target values based on the rewards and the next action values
    next_q_values = torch.zeros((batch_size, 1), device=device).to(device)
    next_q_values[has_next_state] = next_action_values.detach()
    target_q_values = reward_batch + gamma * next_q_values

    ## Now we calculate the loss
    loss = loss_fn( state_q_values, target_q_values ).mean()

    ## Now we optimize the model
    optimiser.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad.clip_grad_norm_(policy_net.parameters(), 10)
    optimiser.step()

    ## We then let the target come slightly closer to the policy
    tau = 1e-3
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_( tau * policy_param.data + ( 1.0 - tau ) * target_param.data )



























