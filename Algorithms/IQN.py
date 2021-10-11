from pathlib import Path
import numpy as np
import numpy.random as rd

import time
import torch as T
import torch.nn as nn

import torch.nn.functional as F
from Resources import utils as myUT
from Resources import memory as myMY
from Resources import networks as myNW

class IQNDuelMLP(nn.Module):
    ''' A duelling mlp for the distributional alorithm IQN
        It is contains a base_stream, a tau embedding stream, and a final stream.
        The final stream has a duelling arcitecture and can be equipped with noisy layers.
    '''
    def __init__(self, n_quantiles, state_shape, n_actions, **network_dict):
        ''' Constructor for IQNDuelMLP
        args:
            n_quantiles: The number of individual quantiles to calculate for each forward pass
            state_shape: The dimensions of the environment's state vector
            n_actions: The number of possible actions for the environment
            network_dict: Arguments for the mlp_blocks creator
        '''
        super().__init__()

        ## Attributes needed in the forward pass
        self.duel = network_dict.pop('duel')
        self.n_actions = n_actions

        ## The various dictionaries for each MLP
        base_kwargs = network_dict.copy()
        joined_kwargs = network_dict.copy()
        joined_kwargs['depth'] =2
        AV_kwargs = network_dict.copy()
        AV_kwargs['depth'] = 1

        ## The IQN parameters
        self.n_quantiles = n_quantiles
        self.quant_emb_dim = 64
        self.pi_vals = np.pi * T.arange(1, self.quant_emb_dim+1, dtype=T.float32)

        ## Defining the base stream as an mlp
        self.base_stream = myNW.mlp_blocks(n_in=state_shape, n_out=0, **base_kwargs)

        ## Defining the tau embedding stream as a 2 layer mlp, combined with cosine embedding later
        self.joined_stream = myNW.mlp_blocks(n_in=self.quant_emb_dim+base_kwargs['width'], n_out=0, **joined_kwargs)

        ## Defining the dueling network arcitecture
        self.A_stream = myNW.mlp_blocks(n_in=joined_kwargs['width'], n_out=n_actions, **AV_kwargs)
        if self.duel:
            self.V_stream = myNW.mlp_blocks(n_in=joined_kwargs['width'], n_out=1, **AV_kwargs)

        ## Moving the network to the device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        self.pi_vals = self.pi_vals.to(self.device)

    def tau_embedding(self, b_size, n_quant):
        ''' Generate taus and embed them using a cosine layer
        args:
            b_size: The batch size of the sample, used to calculate how many taus are required
        returns:
            taus: The tau values, needed later for loss calculation
            cos_embd: The cosine embedded values

        '''
        n_taus = b_size * n_quant
        taus = T.rand((n_taus, 1), dtype=T.float32, device=self.device)
        cos_embd = T.cos(self.pi_vals * taus)

        ## Save the taus as class attributes, needed for loss calculation
        self.taus = taus.view(b_size, n_quant)

        return cos_embd

    def forward(self, state, n_quant=None):
        ''' Calculates the quantile function of the reward at a set number of quantiles
        args:
            state: The input batched state vector
        returns:
            A: The action (advantage) values of shape (batch, n_act, n_quant)
            n_quant: The number of quantiles to calculate, default points to class attribute
        '''

        if n_quant is None:
            n_quant = self.n_quantiles

        ## Generate the taus and get their embeddings
        batch_size = state.shape[0]
        cos_embd = self.tau_embedding(batch_size, n_quant)

        ## We pass the state through the base layer
        base_out = self.base_stream(state)

        ## For each state we need n quantile outputs, to do that we need to pass it through the next part of the network n times
        ## So we duplicate each state embedding n times along the batch dimension
        rpt_base_out = T.repeat_interleave(base_out, n_quant, dim=0)

        ## Combine the state embedding and the cosine embedding of the taus
        joined = T.cat([rpt_base_out, cos_embd], dim=-1)

        ## Pass through the joined stream
        joined_out = self.joined_stream(joined)

        ## Now we feed the joined information through the duelling streams
        ## Breaking up each sample into (batch, n_act, n_quant)
        A = self.A_stream(joined_out).view(batch_size, n_quant, self.n_actions).transpose(-1, -2)
        if self.duel:
            V = self.V_stream(joined_out).view(batch_size, n_quant, 1).transpose(-1, -2)
            A = V + A - A.mean(dim=1, keepdim=True)

        return A

    def save_checkpoint(self, flag=""):
        print("... saving network checkpoint ..." )
        T.save(self.state_dict(), self.chpt_file+flag)

    def load_checkpoint(self, flag=""):
        print("... loading network checkpoint ..." )
        self.load_state_dict(T.load(self.chpt_file+flag))


class IQN_Agent(object):
    ''' An implicit quantile network deep Q learning agent with an experience replay for use in discrete environments
    '''
    def __init__(self,
                 name,
                 save_dir,
                 env,
                 alg_kwargs,
                 network_dict,
                 training_dict,
                 exploration_dict,
                 memory_dict):
        ''' Constructor method for IQN_Agent
        args:
            name: The name of the algorithm, used for saving and reloading
            save_dir: The directory where to save the model checkpoints
            env:  A copy of the environment which will be run over, only used to store action and state shapes
            alg_kwargs: A collection of kwargs used to define any algorithm specific parameters
            network_dict: A collection of kwargs used to define the network hyperparameters (depth, width, etc)
            training_dict: A collection of kwargs used to define the training hyperparameters (lr, optim, etc)
            exploration_dict: A collection of kwargs used to define the exploration strategy of the agent (epsilon, etc)
            memory_dict: A collection of kwargs used to define the experience replay (capacity, etc)
        '''

        ## Add the contents of the of dictionaries as class attributes
        self.path = Path(save_dir, name + '_IQN_' + env.unwrapped.spec.id)
        self.__dict__.update(alg_kwargs)
        self.__dict__.update(network_dict)
        self.__dict__.update(training_dict)
        self.__dict__.update(exploration_dict)
        self.__dict__.update(memory_dict)

        ## Save the state and action spaces from the environment
        self.state_shape = env.reset().shape
        self.n_actions = env.action_space.n

        ## Initialise the variables for training and exploration
        self.test_mode = False
        self.learn_step_counter = 0
        self.n_gamma = self.gamma ** self.n_step ## So we dont have to do it every time

        ## Initlialise the policy and target networks and sync them together)
        self.policy_net = IQNDuelMLP(self.n_quantiles, self.state_shape[0], self.n_actions, **network_dict)
        self.target_net = IQNDuelMLP(self.n_quantiles, self.state_shape[0], self.n_actions, **network_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        ## Initialise the optimiser for the policy network and the loss function to use for TD learning
        self.optimiser = myUT.get_opt(self.opt_nm, self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = myUT.get_loss(self.loss_nm)

        ## Initlalise the agent's experience replay buffer
        self.memory = myMY.Experience_Replay(self.state_shape, gamma=self.gamma, **memory_dict)

        ## The lambda function needed because pytorch does not like broadcasting in loss calculations
        self.huber_fn  = lambda x: T.where(x.abs()<1, x*x/2, (x.abs()-0.5))

    def choose_action(self, state):
        ''' Select an action based on an exploring/greedy strategy
        args:
            state: The single state to apply the policy to
        returns:
            The selected action index
            The expected q value of that action using the policy network
        '''

        ## Check if we have exeeded the initial exploration stage
        if not self.test_mode and self.memory.mem_cntr < self.freeze_up:
            action = rd.randint(self.n_actions)
            act_value = 0

        ## Use epsilon greedy action selection
        elif not self.test_mode and rd.random() < self.eps:
                action = rd.randint(self.n_actions)
                act_value = 0
                self.eps = max(self.eps - self.eps_dec, self.eps_min)

        ## Act based on the policy network's q value estimate
        else:
            with T.no_grad():
                state_tensor = T.tensor(state, device=self.policy_net.device, dtype=T.float32)
                pol_dist = self.policy_net(state_tensor.unsqueeze(0), n_quant=32).squeeze()
                Q_values = pol_dist.mean(dim=-1)
                action = T.argmax(Q_values).item()
                act_value = myUT.to_np(Q_values[action])

        return action, act_value

    def store_transition(self, state, action, reward, next_state, done):
        ''' A direct interface to the replay buffer, so that no external class needs to access it
        '''
        self.memory.store_transition(state, action, reward, next_state, done)

    def sync_target_network(self):
        ''' Sync the target network with the policy network using the target_sync attribute
        '''

        ## If target_sync is a fraction it implies exponential moving average
        if self.target_sync < 1:
            with T.no_grad():
                for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    tp.data.copy_(self.target_sync*pp.data + (1.0-self.target_sync)*tp.data)

        ## If target_sync is an inter it implies hard/complete updates
        else:
            if self.learn_step_counter % self.target_sync == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_models(self, flag=''):
        ''' Saves a checkpoint of the target and policy network
        '''
        pass

    def load_models(self, flag=''):
        ''' Loads a checkpoint of the target and policy network
        '''
        pass

    def train(self):

        ## We dont train until the memory is at least one batch_size
        if self.memory.mem_cntr < max(self.batch_size, self.freeze_up):
            return 0

        ## We check if the target network needs to be replaced
        self.sync_target_network()

        ## We zero out the gradients, as required for each pyrotch train loop
        self.optimiser.zero_grad()

        ## Collect the batch
        states, actions, rewards, next_states, dones, is_weights, indices = self.memory.sample_memory(self.batch_size, dev=self.policy_net.device)

        ## The single valued tensors are unsqueezed to allow for broadcasting
        rewards.unsqueeze_(-1)
        dones.unsqueeze_(-1)

        ## We use the range of up to batch_size just for indexing methods
        batch_idxes = list(range(self.batch_size))

        ## To increase the speed of this step we do it without keeping track of gradients
        with T.no_grad():

            ## First we need the next state distribution using the policy network for double Q learning
            pol_dist_next = self.policy_net(next_states)

            ## We then find the Q-values of the actions by summing over the supports
            pol_Q_next = T.sum(pol_dist_next, dim=-1)

            ## Can then determine the optimum actions
            next_actions = T.argmax(pol_Q_next, dim=1)

            ## We now use the target network to get the distributions of these actions
            tar_dist_next = self.target_net(next_states)[batch_idxes, next_actions]

            ## We can then find the new supports using the distributional Bellman Equation
            td_target_dist = rewards + self.n_gamma * tar_dist_next * (~dones)
            td_target_dist = td_target_dist.detach()

        ## Now we calculate the network estimates for the state and actions from the memory
        pol_dist = self.policy_net(states)[batch_idxes, actions]

        ## To calculate the quantile regression loss we construct matrices using samples x targets
        diff_matrix = td_target_dist.unsqueeze(-1) - pol_dist.unsqueeze(-1).transpose(1,2)

        ## Broadcast the taus to match the matrix shape, but with the same values in each column
        taus = self.policy_net.taus.unsqueeze(-2).expand(-1, self.n_quantiles, self.n_quantiles)

        ## We then find the loss function using the QR equation
        QRloss = self.huber_fn(diff_matrix) * (taus - (diff_matrix.detach()>0).float()).abs()

        ## We then need to find the mean for each matrix
        QRloss = QRloss.mean(dim=(-1, -2))

        ## Use this loss as new errors to be used in PER and update the replay
        if self.PER_on:
            new_errors = QRloss.detach().cpu().squeeze()
            self.memory.update_prioties(indices, new_errors)

        ## Now we use the loss for graidient desc, applying is weights if using PER
        loss = QRloss * is_weights.unsqueeze(1)
        loss = loss.mean()
        loss.backward()

        ## We might want to clip the gradient before performing SGD
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimiser.step()

        self.learn_step_counter += 1

        return loss.item()
