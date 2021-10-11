import torch as T
from collections import deque

class Experience_Replay:
    ''' The standard experience replay buffer for off-policy models (q-learning)
    An experience replay stores memories or experiences for the agent to reveiw and learn from.
    A single experience is defined as a tuple of (state, action, reward, next_state, done).
    All the variables needed for the q-learning update algorithm.
     - Can also be configured as a priority experience replay,
       with extra weights that sample inaccurate events with higher probability
     - Can also be configured as a n-step experience replay,
       where the memories are state, action, n x rewards, next_state after n
    '''
    def __init__(self, state_shape, act_shape=None, gamma=None, **mem_kwargs):
        ''' Constructor class for Experience_Replay
        Creates numpy arrays for each part of the memory tuple
        args:
            state_shape: The the shape of the vector used in the state/next_state memory
        kwargs:
            act_shape: The shape of the action space (for continuous environments)
                       If None it assumes actions are discrete and thus only the index is stored
            gamma:     The future discount factor, needed in n_step returns
            mem_kwargs: Other memory kwargs that come from the config file
        '''

        ## Save the options as class attributes
        self.__dict__.update(mem_kwargs)
        self.gamma = gamma
        self.mem_cntr = 0

        ## To allow for discrete or continuous actions
        self.is_cont = (act_shape is not None)
        act_type  = T.float32 if self.is_cont else T.int64
        act_shape = (self.capacity, *act_shape) if self.is_cont else self.capacity

        ## The actual memory arrays which will hold the data
        self.state_memory      = T.zeros((self.capacity, *state_shape), dtype=T.float32)
        self.action_memory     = T.zeros(act_shape, dtype=act_type)
        self.reward_memory     = T.zeros(self.capacity, dtype=T.float32)
        self.next_state_memory = T.zeros_like(self.state_memory)
        self.terminal_memory   = T.zeros(self.capacity, dtype=T.bool)

        ## For PER we also need to store a tensor of the weights for each sample
        if self.PER_on:
            self.p_max = 1
            self.tot_priority = 0
            self.priorities = T.zeros(self.capacity, dtype=T.float32)

        ## For n step learning we use a deque as a buffer
        if self.n_step > 1:
            self.n_step_buffer = deque(maxlen=self.n_step)

    def store_transition(self, state, action, reward, next_state, done):
        ''' Stores the memory at the location defined by the index
        '''

        ## Place the transition inside a small n_step_buffer
        if self.n_step > 1:
            self.n_step_buffer.append(tuple([state, action, reward, next_state, done]))
            if len(self.n_step_buffer) < self.n_step and not done:
                return ## Do nothing until the buffer is full
            state, action, reward, next_state, done = self._get_n_step_info()

        ## Check what index is being filled/replaced
        index = self.mem_cntr % self.capacity

        ## Store the transition information in their respective tensors
        self.state_memory[index]      = T.from_numpy(state)
        self.action_memory[index]     = T.from_numpy(action) if self.is_cont else action
        self.reward_memory[index]     = reward
        self.next_state_memory[index] = T.from_numpy(next_state)
        self.terminal_memory[index]   = done

        ## For PER, new states are stored with maximum priority and the total is updated
        if self.PER_on:
            self.tot_priority += self.p_max - self.priorities[index]
            self.priorities[index] = self.p_max

        self.mem_cntr += 1

    def sample_memory(self, batch_size, dev='cpu'):
        ''' Loads memories by randomly sampling the replay buffer
        args:
            batch_size: The number of memories to load as a batch
        kwargs:
            dev:        The device to move the tensors to
        returns:
            states, ... : batched forms of each part of the memory tuple
            is_weights:   importance sampling weights
            indices:      indices of the memories returned
        '''

        max_mem = min(self.mem_cntr, self.capacity)

        ## Calculate the priorities for each memory
        if self.PER_on:
            priorities = self.priorities[:max_mem]
        else:
            priorities = T.ones(max_mem)

        ## Generate indices using the pytorch multinomial weighted sampling method
        indices = T.multinomial(priorities, batch_size, replacement=True)

        ## Collect the data using those indices
        states      = self.state_memory[indices].to(dev)
        actions     = self.action_memory[indices].to(dev)
        rewards     = self.reward_memory[indices].to(dev)
        next_states = self.next_state_memory[indices].to(dev)
        dones       = self.terminal_memory[indices].to(dev)

        ## Calculate the importance sampling weights using priorities
        if self.PER_on:
            is_weights = (max_mem*priorities[indices]/self.tot_priority)**(-self.beta) ## Calculate biased IS weights
            is_weights *= batch_size/T.sum(is_weights) ## Normalise so each batch has same sum of weights
            self.beta = min([1.0, self.beta + self.beta_inc]) ## Update the bias parameter
        else:
            is_weights = T.ones_like(indices)

        ## Move importance sampling weights to device
        is_weights = is_weights.to(dev)

        return states, actions, rewards, next_states, dones, is_weights, indices

    def update_prioties(self, indices, errors):
        ''' Updates the tensor holding the event priorities
        args:
            indices: The locations of the events to update
            errors: The td errors from which we can calculate the new priorities
        '''

        ## Calculate the new priorities
        new_priors = errors ** self.alpha

        ## Update the total, the old and the maximum
        self.priorities[indices] = new_priors
        self.p_max = T.max(self.priorities)
        self.tot_priority = T.sum(self.priorities)

    def _get_n_step_info(self):
        ''' Returns the n step transition of the current buffer
        '''

        ## The state and action for n_step are taken from the start of the buffer
        n_state    = self.n_step_buffer[0][0]
        n_action   = self.n_step_buffer[0][1]

        ## The next state and terminal signal are taken from the end of the buffer
        n_next_state = self.n_step_buffer[-1][3]
        n_done       = self.n_step_buffer[-1][4]

        ## To calculate the reward we iterate backwards towards the begining adding them up
        n_reward = 0
        for transition in reversed(list(self.n_step_buffer)):
            rew = transition[2]
            n_s = transition[3]
            don = transition[4]

            ## Update our reward with gamma annealing
            n_reward = rew + self.gamma * n_reward

            ## If the transition lead to a terminal state
            ## We shift the n_step to be shorter
            if don:
                n_next_state  = n_s
                n_done        = don
                n_reward      = rew

        return n_state, n_action, n_reward, n_next_state, n_done
