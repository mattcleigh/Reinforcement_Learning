import numpy as np
from collections import deque

class SmallMemory:
    """ This class is no longer in use since switching to vectorised workers

        This is similar to the DQN experience replay and
        exists to decorrelate updates. There are N workers,
        and each are allowed to expereience F frames. So the memory
        holds NxF transitions for the update.
        All transitions are pulled each call and all are replaced before the next.
    """

    def __init__(self, capacity, state_input_shape):

        ## Descriptions of the memory
        self.capacity  = capacity

        ## The actual memory arrays which will hold the data
        self.states  = np.zeros( (capacity, *state_input_shape), dtype=np.float32 )
        self.actions = np.zeros(  capacity, dtype=np.int64 )
        self.values  = np.zeros(  capacity, dtype=np.float32 )

    def reset(self):
        self.mem_cntr = 0
        self.states.fill(0)
        self.actions.fill(0)
        self.values.fill(0)

    def store_transition(self, state, action, value):

        ## We store the transition information in its respective arrays
        self.states[self.mem_cntr]  = state
        self.actions[self.mem_cntr] = action
        self.values[self.mem_cntr]  = value
        self.mem_cntr += 1



class SumTree(object):
    """ A binary tree object where each node contains the sum
        of it's children. Each leaf node will contain the
        probability of being sampled. Only returns the index of
        the sample, not the data itself.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.n_entries = 0
        self.tree = np.zeros( 2 * capacity - 1 )

    def add(self, priority, data_index):

        ## We call the update function to make changes to the tree
        self.update(priority, data_index)

        ## Update the number of entries in the tree
        if self.n_entries < self.capacity:
            self.n_entries += 1


    def update(self, priority, data_index):

        ## We need to convert to the index of the leaf node
        tree_index = data_index + self.capacity - 1

        ## We calculate the change in priorities, and replace
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        ## Propagate the change up through the tree
        while tree_index != 0:
            tree_index = (tree_index-1) // 2
            self.tree[tree_index] += change

    def get_index(self, value):

        ## Find the index of the sample whos segment containings a value
        parent_index = 0
        while True:
            left_child_index  = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value = value - self.tree[left_child_index]
                    parent_index = right_child_index

        priority   = self.tree[leaf_index]
        data_index = leaf_index - self.capacity + 1

        return data_index, priority

    @property
    def total_priority(self):
        return self.tree[0]


class Experience_Replay(object):
    """ A class which contains the standard experience replay buffer for DQN models
    """
    def __init__(self, capacity, state_input_shape):

        ## Descriptions of the memory
        self.capacity  = capacity
        self.mem_cntr  = 0

        ## The actual memory arrays which will hold the data
        self.state_memory      = np.zeros( (capacity, *state_input_shape), dtype=np.float32 )
        self.next_state_memory = np.zeros( (capacity, *state_input_shape), dtype=np.float32 )
        self.action_memory     = np.zeros(  capacity, dtype=np.int64   )
        self.reward_memory     = np.zeros(  capacity, dtype=np.float32 )
        self.terminal_memory   = np.zeros(  capacity, dtype=np.bool    )

    def store_transition(self, state, action, reward, next_state, done):

        ## We check what index is being filled/replaced
        index = self.mem_cntr % self.capacity

        ## We store the transition information in its respective arrays
        self.state_memory[index]      = state
        self.action_memory[index]     = action
        self.reward_memory[index]     = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index]   = done
        self.mem_cntr += 1

    def sample_memory(self, batch_size):

        ## We collect the indices using a uniform sample
        max_mem = min(self.mem_cntr, self.capacity)
        indices = np.random.choice(max_mem, batch_size, replace=False)

        ## We then collect the data using those indices
        states      = self.state_memory[indices]
        actions     = self.action_memory[indices]
        rewards     = self.reward_memory[indices]
        next_states = self.next_state_memory[indices]
        dones       = self.terminal_memory[indices]

        ## The PER returns is_weights and indicies so to
        ## unify these methods we return them too, even though
        ## we do not use them.
        return states, actions, rewards, next_states, dones, [1], indices

class PER(object):
    """ A class which contains the prioritised experience replay buffer for DQN models
        using the sumtree method of sampling with values
    """
    def __init__(self, capacity, state_input_shape, eps=0.01, a=0.5, beta=0.4, beta_inc=1e-4, max_priority=1):

        ## Descriptions of the memory
        self.capacity  = capacity
        self.mem_cntr  = 0

        ## The actual memory arrays which will hold the data
        self.state_memory      = np.zeros( (capacity, *state_input_shape), dtype=np.float32 )
        self.next_state_memory = np.zeros( (capacity, *state_input_shape), dtype=np.float32 )
        self.action_memory     = np.zeros(  capacity, dtype=np.int64   )
        self.reward_memory     = np.zeros(  capacity, dtype=np.float32 )
        self.terminal_memory   = np.zeros(  capacity, dtype=np.bool    )

        ## The extra features needed for priotitised sampling
        self.eps       = eps
        self.a         = a
        self.beta      = beta
        self.beta_inc  = beta_inc
        self.max_priority = max_priority
        self.sumtree   = SumTree(capacity)

    def store_transition(self, state, action, reward, next_state, done):

        ## We check what index is being filled/replaced
        data_index = self.mem_cntr % self.capacity

        ## We store the transition information in its respective arrays
        self.state_memory[data_index]      = state
        self.action_memory[data_index]     = action
        self.reward_memory[data_index]     = reward
        self.next_state_memory[data_index] = next_state
        self.terminal_memory[data_index]   = done
        self.mem_cntr += 1

        ## We then store the max_priority into the sumtree in the corresponding location
        self.sumtree.add( self._get_priority(), data_index )

    def _get_priority(self, error=None):
        if error is None:
            priority = self.max_priority
        else:
            priority = ( np.abs(error) + self.eps ) ** self.a
            self.max_priority = max(self.max_priority, priority )
        return priority

    def sample_memory(self, batch_size):

        ## Sample indices using the sumtree which also gives us the IS weights
        is_weights, indices = self._sample_indices( batch_size )

        ## We then collect the data using those indices
        states      = self.state_memory[indices]
        actions     = self.action_memory[indices]
        rewards     = self.reward_memory[indices]
        next_states = self.next_state_memory[indices]
        dones       = self.terminal_memory[indices]

        ## Update beta
        self.beta = np.min([1., self.beta + self.beta_inc])

        return states, actions, rewards, next_states, dones, is_weights, indices

    def _sample_indices(self, batch_size):

        ## We need a list to fill with indices, and a segment length
        indices   = []
        priorities = []
        seg_length = self.sumtree.total_priority / (batch_size)

        ## Collect a new sample of data from the tree
        for i in range(batch_size):

            value = np.random.random() * seg_length + seg_length * i
            data_index, priority = self.sumtree.get_index(value)

            indices.append( data_index )
            priorities.append( priority )

        ## Convert to numpy arrays
        indices   = np.array( indices,   dtype = np.int64 )
        priorities = np.array( priorities, dtype = np.float32 )

        ## We need the importance sampling weights from the sumtree as well
        probs = priorities / self.sumtree.total_priority
        is_weights = np.power(self.sumtree.n_entries * probs, -self.beta )
        is_weights /= is_weights.max()

        return is_weights, indices

    def batch_update(self, indices, new_errors):

        for idx, err in zip(indices, new_errors):
            priority = self._get_priority(err)
            self.sumtree.update(priority, idx)

    @property
    def n_entries(self):
        return self.sumtree.n_entries


class N_Step_PER(object):
    """ A class which contains the prioritised experience replay buffer for DQN models using
        the sumtree method of sampling with values, also equipped for multiple timestep learning.
    """
    def __init__(self, capacity, state_input_shape, eps=0.01, a=0.5, beta=0.4, beta_inc=1e-4, max_priority=1, n_step=3, gamma=0.999):

        ## Descriptions of the memory
        self.capacity  = capacity
        self.mem_cntr  = 0

        ## The actual memory arrays which will hold the data
        self.state_memory      = np.zeros( (capacity, *state_input_shape), dtype=np.float32 )
        self.next_state_memory = np.zeros( (capacity, *state_input_shape), dtype=np.float32 )
        self.action_memory     = np.zeros(  capacity, dtype=np.int64   )
        self.reward_memory     = np.zeros(  capacity, dtype=np.float32 )
        self.terminal_memory   = np.zeros(  capacity, dtype=np.bool    )

        ## The extra features needed for priotitised sampling
        self.eps       = eps
        self.a         = a
        self.beta      = beta
        self.beta_inc  = beta_inc
        self.max_priority = max_priority
        self.sumtree   = SumTree(capacity)

        ## For N-Step Learning
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)
        self.gamma = gamma

    def store_transition(self, state, action, reward, next_state, done):

        ## Place the transition inside the n_step_buffer and dont store until fill
        transition = tuple( [state, action, reward, next_state, done] )
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step and not done:
            return

        ## Produce the full n_step transition
        n_state, n_action, n_reward, n_next_state, n_done = self._get_n_step_info()

        ## We check what index is being filled/replaced
        data_index = self.mem_cntr % self.capacity

        ## Store the n_step transition data in its respecive arrays
        self.state_memory[data_index]      = n_state
        self.action_memory[data_index]     = n_action
        self.reward_memory[data_index]     = n_reward
        self.next_state_memory[data_index] = n_next_state
        self.terminal_memory[data_index]   = n_done
        self.mem_cntr += 1

        ## We then store the max_priority into the sumtree in the corresponding location
        self.sumtree.add( self._get_priority(), data_index )

    def _get_priority(self, error=None):
        if error is None:
            priority = self.max_priority
        else:
            priority = ( np.abs(error) + self.eps ) ** self.a
            self.max_priority = max(self.max_priority, priority )
        return priority

    def _get_n_step_info(self):

        ## The state and action for n_step is the same
        n_state    = self.n_step_buffer[0][0]
        n_action   = self.n_step_buffer[0][1]

        ## We collect the last elements of our buffer for the n_step
        n_next_state = self.n_step_buffer[-1][3]
        n_done       = self.n_step_buffer[-1][4]
        n_reward     = 0

        ## iterate backwards towards the begining
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


    def sample_memory(self, batch_size):

        ## Sample indices using the sumtree which also gives us the IS weights
        is_weights, indices = self._sample_indices( batch_size )

        ## We then collect the data using those indices
        states      = self.state_memory[indices]
        actions     = self.action_memory[indices]
        rewards     = self.reward_memory[indices]
        next_states = self.next_state_memory[indices]
        dones       = self.terminal_memory[indices]

        ## Update beta
        self.beta = np.min([1., self.beta + self.beta_inc])

        return states, actions, rewards, next_states, dones, is_weights, indices

    def _sample_indices(self, batch_size):

        ## We need a list to fill with indices, and a segment length
        indices   = []
        priorities = []
        seg_length = self.sumtree.total_priority / (batch_size)

        ## Collect a new sample of data from the tree
        for i in range(batch_size):

            value = np.random.random() * seg_length + seg_length * i
            data_index, priority = self.sumtree.get_index(value)

            indices.append( data_index )
            priorities.append( priority )

        ## Convert to numpy arrays
        indices    = np.array( indices,    dtype = np.int64 )
        priorities = np.array( priorities, dtype = np.float32 )

        ## We need the importance sampling weights from the sumtree as well
        probs = priorities / self.sumtree.total_priority
        is_weights = np.power(self.sumtree.n_entries * probs, -self.beta )
        is_weights /= is_weights.max()

        return is_weights, indices

    def batch_update(self, indices, new_errors):

        for idx, err in zip(indices, new_errors):
            priority = self._get_priority(err)
            self.sumtree.update(priority, idx)

    @property
    def n_entries(self):
        return self.sumtree.n_entries


class Cont_Exp_Replay(Experience_Replay):
    """ A class which contains the standard experience replay buffer for continuous action spaces.
        This differs from the usual object as now each action is a full vector
        containing floats, rather than a single int object
    """
    def __init__(self, capacity, state_input_shape, n_actions):
        super(Cont_Exp_Replay, self).__init__(capacity,  state_input_shape)
        self.action_memory = np.zeros( (capacity, n_actions), dtype=np.float32 )

class Cont_PER(PER):
    """ A class which contains the PER replay buffer for continuous action spaces.
        This differs from the usual object as now each action is a full vector
        containing floats, rather than a single int object
    """
    def __init__(self, capacity, state_input_shape, n_actions, eps=0.01, a=0.5, beta=0.4, beta_inc=1e-4, max_priority=1):
        super(Cont_PER, self).__init__(capacity, state_input_shape, eps, a, beta, beta_inc, max_priority)
        self.action_memory = np.zeros( (capacity, n_actions), dtype=np.float32 )

class Cont_N_Step_PER(N_Step_PER):
    """ A class which contains the n-step PER replay buffer for continuous action spaces.
        This differs from the usual object as now each action is a full vector
        containing floats, rather than a single int object
    """
    def __init__(self, capacity, state_input_shape, n_actions, eps=0.01, a=0.5, beta=0.4, beta_inc=1e-4, max_priority=1, n_step=3, gamma=0.99):
        super(Cont_N_Step_PER, self).__init__(capacity, state_input_shape, eps, a, beta, beta_inc, max_priority, n_step, gamma)
        self.action_memory = np.zeros( (capacity, n_actions), dtype=np.float32 )
