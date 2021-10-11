from pathlib import Path
import numpy as np
import numpy.random as rd

import torch as T
import torch.nn as nn

from Resources import utils as myUT
from Resources import memory as myMY
from Resources import networks as myNW

class DuelMLP(nn.Module):
    ''' A simple duelling mlp for the D3QN algorithm.
    '''
    def __init__(self, state_shape, n_actions, **network_dict):
        ''' Constructor for DuelMLP
        args:
            state_shape: The dimensions of the environment's state vector
            n_actions: The number of possible actions for the environment
            network_dict: Arguments for the mlp_blocks creator
        '''
        super().__init__()

        ## Save the duelling option for the forward pass
        self.duel = network_dict.pop('duel')
        base_kwargs = network_dict.copy()
        AV_kwargs = network_dict.copy()
        AV_kwargs['depth'] = 1

        ## Defining the base and action-advantage stream of the informaiton
        self.base_stream = myNW.mlp_blocks(n_in=state_shape, n_out=0, **base_kwargs)
        self.A_stream = myNW.mlp_blocks(n_in=base_kwargs['width'], n_out=n_actions, **AV_kwargs)

        ## If the network employs a duelling architecture, then it also has a state value stream
        if self.duel:
            self.V_stream = myNW.mlp_blocks(n_in=base_kwargs['width'], n_out=1, **AV_kwargs)

        ## Moving the network to the device
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):

        ## Pass the state information through both the base and action stream
        shared_out = self.base_stream(state)
        A = self.A_stream(shared_out)

        ## If duelling, then add the value stream and recalibrate
        if self.duel:
            V = self.V_stream(shared_out)
            A = A + V - A.mean(dim=-1, keepdim=True)

        return A

class D3QN_Agent:
    ''' A (Double-Duelling) Deep Q learning agent with an experience replay for use in discrete environments
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
        ''' Constructor method for D3QN_Agent
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
        self.path = Path(save_dir, name + '_D3QN_' + env.unwrapped.spec.id)
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
        self.policy_net = DuelMLP(self.state_shape[0], self.n_actions, **network_dict)
        self.target_net = DuelMLP(self.state_shape[0], self.n_actions, **network_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        ## Initialise the optimiser for the policy network and the loss function to use for TD learning
        self.optimiser = myUT.get_opt(self.opt_nm, self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = myUT.get_loss(self.loss_nm)

        ## Initlalise the agent's experience replay buffer
        self.memory = myMY.Experience_Replay(self.state_shape, gamma=self.gamma, **memory_dict)

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
                Q_values = self.policy_net(state_tensor)
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
        ''' The main training step which is called after enough memories have been stored into the experience replay
        - The memories are sampled as a batch
        - The best actions for the next states are determined using the policy network (double q learning)
        - The values of those actions are calculated using the target network
        - The TD target is calculated using the equation (R + gamma^n * Q(ns, na*))
        - The policy network estimate is calculated
        - The loss is applied between the two (with additional IS weights as required)
        - The PER weights and the policy net is updated using the TD error
        '''
        ## We dont train until the memory is at least one batch_size
        if self.memory.mem_cntr < max(self.batch_size, self.freeze_up):
            return 0

        ## We check if the target network needs to be replaced
        self.sync_target_network()

        ## We zero out the gradients, as required for each pyrotch train loop
        self.optimiser.zero_grad()

        ## Collect the batch
        states, actions, rewards, next_states, dones, is_weights, indices = self.memory.sample_memory(self.batch_size, dev=self.policy_net.device)

        ## We use the range of up to batch_size just for indexing methods (simpler than using torch.gather)
        batch_idxes = list(range(self.batch_size))

        ## To increase the speed of this step we do it without keeping track of gradients
        with T.no_grad():

            ## First we find the Q-values of the next states
            pol_Q_next = self.policy_net(next_states)

            ## Can then determine the optimum actions
            next_actions = T.argmax(pol_Q_next, dim=1)

            ## We now use the target network to get the values of these actions
            tar_Q_next = self.target_net(next_states)[batch_idxes, next_actions]

            ## Calculate the target values based on the Bellman Equation
            td_target = rewards + self.n_gamma * tar_Q_next * (~dones)
            td_target = td_target.detach()

        ## Now we calculate the network estimates for the state and actions from the memory
        pol_Q = self.policy_net(states)[batch_idxes, actions]

        ## Calculate the TD-Errors to be used in PER and update the replay
        if self.PER_on:
            new_errors = T.abs(pol_Q - td_target).detach().cpu().squeeze()
            self.memory.update_prioties(indices, new_errors)

        ## Now we use the loss for graidient desc, applying is weights from PER
        loss = self.loss_fn(pol_Q, td_target) * is_weights.unsqueeze(1)
        loss = loss.mean()
        loss.backward()

        ## We might want to clip the gradient before performing SGD
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimiser.step()

        self.learn_step_counter += 1

        return loss.item()
