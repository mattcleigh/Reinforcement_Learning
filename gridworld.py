import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

from itertools import count
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import Resources.utils as myUT

rtg=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256)

class GridWorldEnv:
    '''A simple grid based reinforcement learning environment, primarily for demonstration perposes.
    The environment keeps track of the current state and updates it by receiving actions from an agent,
    and dishes out rewards based on the encounterd new states.
    We can also use the environment render function to allow us to observe the episode
    '''
    def __init__(self, width=8, height=8,
                       rnd_strt=False,
                       stp_rwrd=0,
                       chs_rwrd=10, n_chs=1,
                       cat_rwrd=-10, n_cat=10, cat_kill=True):
        ''' The constructor method for GridWorldEnv
        kwargs:
           width:    The number of grid squares in the x direction
           height:   The number of grid squares in the y direction
           rnd_strt: If True, the current state at the begining of the episode is chosen randomly
                     If False, the current state will always be in the top left-hand corner
           stp_rwrd: The reward given for taking any action (setting to small negative gives urgency)
           chs_rwrd: The reward given for finding the cheese (also a winning state which ends the episode)
           cat_rwrd: The reward/punishment given for running into a cat (meow)
           cat_kill: True if running into cat ends the episode
                     - If False the agent will 'bounce off' while still incurring the penalty
        '''

        ## Define the possible actions for the gridworld, which is just the 4 cardinal directions
        self.actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])

        ## Save the dimensions of the gridworld (used for checking out of bounds)
        self.shape = np.array([width, height])

        ## Save some of the configurations which will be used
        self.stp_rwrd = stp_rwrd
        self.rnd_strt = rnd_strt
        self.cat_kill = cat_kill

        ## Stoe a list of every possible state in the gridworld
        self.all_states = np.array(np.unravel_index(np.arange(np.prod(self.shape)), self.shape)).T

        ## Place the winning (cheese) and loosing (cat) states into the grid (cant be on the first corner)
        end_idxes = rd.choice(len(self.all_states)-1, size=n_chs+n_cat, replace=False) + 1
        self.end_states = self.all_states[end_idxes]
        self.chs_states = self.end_states[:n_chs]
        self.cat_states = self.end_states[n_chs:]

        ## Calculate the reward grid for quick and batched reward calculations
        self.r_grid = np.zeros((width, height)) + stp_rwrd
        self.r_grid[tuple(self.chs_states.T)] += chs_rwrd
        self.r_grid[tuple(self.cat_states.T)] += cat_rwrd

        ## Call reset (which defines the current state)
        self.state = np.array([0,0])
        self.reset()

    def reset(self):
        ''' Resets the location of the mouse for the start of a new episode
        '''

        ## If we want to randomly place the mouse somewhere we cant start on an end state
        if self.rnd_strt:
            valid_start = False
            while not valid_start:
                self.state = rd.randint(low=[0,0], high=self.shape)
                valid_start = (not myUT.isin(self.state, self.cat_states) and (not myUT.isin(self.state, self.chs_states)))

        ## Otherwise we always start in the bottom corner (this by design will never be an end state)
        else:
            self.state = np.array([0,0])

        ## We also need to reset the history of the mouse (for rendering)
        self.ms_history = [self.state]

        ## Reset must return the state to allow for episode initialisation
        return self.state

    def step(self, action, query_state=None):
        ''' Advances the environment by one timestep using an action signal sent by the appropriate agent
        args:
            action: This is an interger which gets decoded into a np.array of the direction to move on the gird
                    For example: [-1,0] = LEFT, [0,1] = UP, etc
        kwargs:
            query_state: Query a paticular state. Note that this will NOT update the internal environment.
                         This is useful for use with dynamic programming methods which solve the environment before
                         actually playing.
        returns:
            next_state: The new locaiton on the grid for the agent/mouse
            reward: The reward/punishment signal for encountering the next_state
            done: A boolean which is True if the next_state is an end_state, indicating the end of an episode
        '''

        ## Move the envorinment to some arbitrary state
        state = self.state if query_state is None else query_state

        ## So far there is no reason to end the episode
        done = False

        ## Decode the action
        action = self.actions[action]

        ## Move the current state by the direction specified by the action (prevent out of bounds)
        state = np.clip(state + action, [0,0], self.shape-1)

        ## Calculate the reward for being an the new state
        reward = self.r_grid[tuple(state)]

        ## Check if the agent has found the cheese
        if myUT.isin(state, self.chs_states):
            done = True

        ## Check if the agent has found a cat
        elif myUT.isin(state, self.cat_states):
            if self.cat_kill:
                done = True ## End the episode
            else:
                state -= action ## Bounce off the cat

        ## Update the environment with the new state only if we are not querying
        if query_state is None:
             self.state = state

        return state, reward, done

    def render(self):
        ''' Renders the gridworld and its inhabitants using matplotlib.
        The first time the render method is called we must setup the figure, place the static cats, etc
        Every subsequent call to render will be a quick update to the mouse position
        '''

        ## First time setup
        if not hasattr(self, 'fig'):

            ## Create the interactive matplotlib figure and axis
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 1, figsize = (5,5))
            self.ax.set_aspect('equal', adjustable='box')

            ## Matching the dimensions of the plot to the gridworld
            self.ax.set_xlim([0, self.shape[0]])
            self.ax.set_ylim([0, self.shape[1]])

            ## Setting up the axis without any ticks or labels
            self.fig.tight_layout()
            self.ax.xaxis.set_ticklabels([])
            self.ax.yaxis.set_ticklabels([])
            self.ax.xaxis.set_ticks_position('none')
            self.ax.yaxis.set_ticks_position('none')

            ## Drawing in the gridlines to indicate the different tiles/states
            self.ax.xaxis.set_major_locator(MultipleLocator(1))
            self.ax.yaxis.set_major_locator(MultipleLocator(1))
            self.ax.grid(True)

            ## Placing the static scatters, the location of the cats and the cheese
            m_size = 15 * 8 / max(self.shape)
            self.ax.plot(*self.cat_states.T + 0.5, 'ro', markersize=m_size)
            self.ax.plot(*self.chs_states.T + 0.5, 'y<', markersize=m_size)

            ## Placing the dynamic scatters, anything to do with the agent/mouse
            self.ms_icon, = self.ax.plot(self.state + 0.5, 'b<', markersize=m_size)
            self.ms_path, = self.ax.plot([], '-b', markersize=m_size)

            ## Flush the canvas
            self.fig.canvas.flush_events()

        ## Update the mouse positions and its history
        self.ms_icon.set_data(self.state + 0.5)
        self.ms_history.append(self.state)
        self.ms_path.set_data(*np.array(self.ms_history).T + 0.5)

        ## Flush the canvas
        self.fig.canvas.flush_events()

class QAgent:
    ''' A Q-Learning agent which can interact with a basic grid world.
    It picks certain actions based on an action value Q(s,a) function.
    Exploitation vs exploitation balance is done through epsilon greedy actions.
    Unlike the dynamic programming methods, this does not need a copy of the environment,
    only the number of states in the gridworld and the action space.
    Using the off policy double Q-learning algorithm (optimal bellman update equation)
    '''
    def __init__(self, env, gamma=0.99, lr=1e-3, eps=1.0, eps_min=0.01, eps_dec=5e-5):
        ''' Constructor method for QAgent
        args:
            env: The environment, only for looking at the state and action space
        kwargs:
            gamma: The discount factor for future rewards
            lr: The learning rate for the updates
            eps: The initial exploration rate for the epsilon greedy policy
            eps_min: The minimum exploration rate
            eps_dec: The amount epsilon decreases every timestep
        '''

        ## Store the gamma, learning rate and epsilon values
        self.gamma = gamma
        self.lr = lr
        self.eps = eps
        self.eps_min = eps_min
        self.eps_dec = eps_dec

        ## The possible actions
        self.n_actions = len(env.actions)

        ## The two action value functions and their mean all start at zeros
        self.act_values = np.zeros((*env.shape, self.n_actions))
        self.act_values_1 = np.zeros((*env.shape, self.n_actions))
        self.act_values_2 = np.zeros((*env.shape, self.n_actions))

        ## For visualisation we need the possible actions, the list of all states and the policy
        self.actions = env.actions
        self.all_states = env.all_states
        self.polviz =  np.zeros((*env.shape, len(self.actions[0])))

        ## The discovered rewards and incoming values per state which is really good for visualisation
        self.rewards = np.zeros(env.shape)
        self.inc_values = np.zeros(env.shape)

    def choose_action(self, state):
        ''' Chooses an action given a state and an epsilon greedy policy.
        Everytime a random action is taken, epsilon is decreased by a small amount.
        args:
            state: The state signal of the environment and input to the Q(s,a) table
        '''

        ## Pick a random action and decrease epsilon
        if rd.random() < self.eps:
            self.eps = max(self.eps-self.eps_dec, self.eps_min)
            action = rd.randint(self.n_actions)
            print(self.eps)

        ## Otherwise pick the greedy option
        else:
            action = np.argmax(self.act_values[tuple(state)])

        return action

    def learn(self, state, action, reward, next_state, done):
        ''' Applies the double q-learning update method to the agent's Q(s,a) table
        args:
            state:  The initial state of the agent
            action: The action the agent took in 'state'
            reward: The reward the agent received after leaving 'state'
            next_state: The next state encountered by the agent
            done: Whether or not the 'next_state' was terminal
        '''

        ## Randomly swap which value table is used for updating and which is used for target
        if rd.random() > 0.5:
            upd_val, trg_val = self.act_values_1, self.act_values_2
        else:
            upd_val, trg_val = self.act_values_2, self.act_values_1

        ## Find the optimal action to take in the next state
        next_act = np.argmax(upd_val[tuple(next_state)])

        ## Get the TD target using the Bellman equation and the next state-action value
        td_target = reward + self.gamma * trg_val[(*next_state, next_act)] * (not done)

        ## Apply the td update to the chosen value function
        upd_val[(*state, action)] += self.lr * (td_target - upd_val[(*state, action)])

        ## Get the average of the two tables for policy and visualisation
        self.act_values[(*state, action)] = 0.5 * ( upd_val[(*state, action)] + trg_val[(*state, action)] )

        ## Update the incoming values for the NEXT STATE for visualisation and the policy
        self.rewards[tuple(next_state)] = reward
        self.inc_values[tuple(next_state)] = reward + self.gamma * np.max(self.act_values[(*next_state, next_act)])
        self.polviz[tuple(state)] = self.actions[np.argmax(self.act_values[tuple(state)])]

        ## Render the state
        self.render(state)

    def render(self, state):
        '''Visualise the experience of the agent as it explors its environment.
        Unlike the DPAgent there is no quiver, just a value grid showing the agent's desire to ENTER a state
        Different from the V(s) which is the return from that state onwards.
        '''

        ## The fist call to render
        if not hasattr(self, 'fig'):
            plt.ion()

            ## Create value/policy figure
            self.fig, self.ax = plt.subplots(1, 1, figsize = (5,5))

            ## Create the values as a grid and the policy as a quiver'
            self.val_grid = self.ax.imshow(self.inc_values.T, origin='lower', vmin=-10, vmax=10, cmap=rtg)
            self.pol_quiver = self.ax.quiver( *self.all_states.T,
                                                  self.polviz[:,:,0], self.polviz[:,:,1],
                                                  pivot='mid', scale=15 )

            ## Create the highlighted box list to append and pop
            rec_dict = {'width':1, 'height':1, 'fc':'none', 'color':'blue', 'linewidth':2}
            self.box = Rectangle((-0.5, -0.5), **rec_dict)
            self.ax.add_patch(self.box)

            self.fig.tight_layout()
            self.ax.axes.xaxis.set_visible(False)
            self.ax.axes.yaxis.set_visible(False)

            ## Flush the canvas
            self.fig.canvas.flush_events()

        ## Update the data and the colours in the value grid
        self.val_grid.set_data(self.inc_values.T)
        self.val_grid.set_clim(np.min(self.inc_values), np.max(self.inc_values))
        self.box.set_xy(tuple(state-0.5))
        self.pol_quiver.set_UVC(self.polviz[:,:,0], self.polviz[:,:,1])

        ## Flush the canvas
        self.fig.canvas.flush_events()

class DPAgent:
    ''' A dynamic programming agent which can solve an entire environment using policy/value iteration.
    Once solved it picks actions based on the state value funciton, it can work out which state it lands up in
    as it has a copy of the environment and can query the state transition funcion (step())
    Currently it supports:
     - Policy Iteration: Cycles of policy evaluation and improvement, good for demonstration purposes
     - Value Iteration: A built in scheme that does both evaluation and improvement in one go
    '''
    def __init__(self, env, gamma=0.99):
        ''' Constructor method for DPAgent
        args:
            env: The environment this agent will act on. This is required as dynamic programming requires
                 access a copy of the state transition matrix (the step() function) to solve for the values
        kwargs:
            gamma: The discount factor for future rewards
        '''

        ## Save a copy of the environment
        self.env = env

        ## Store the gamma factor
        self.gamma = gamma

        ## The initial value and incoming value functions start at zero
        self.values = np.zeros(env.shape)
        self.inc_values = np.zeros(env.shape)

        ## The initial policy starts off as random, also initiate the polviz, for drawing arrows describing the policy
        self.policy = rd.randint(len(env.actions), size=env.shape)
        self.polviz = env.actions[self.policy]

    def update_policy(self, state=None):
        ''' Redefine the agent's current policy to be greedy with respect to the calculated incoming value function.
        Should be called be called after a policy evaluation step or after value iteraction completes
        kwargs:
            state: If given then it only updates the policy on that state, otherwise it does the whole grid
        '''

        ## Which states are to be updated
        updated = self.env.all_states if state is None else [state]

        ## Cycle through all possible states in the gridworld
        for state in updated:

            ## Calculate the action values for each possible action using the next states they lead to
            next_states = np.array([self.env.step(act, query_state=state)[0] for act in np.arange(len(self.env.actions))])
            returns = self.inc_values[tuple(next_states.T)]

            ## Define the policty using best action that maximises this return
            self.policy[tuple(state)] = np.argmax(returns)
            self.polviz[tuple(state)] = self.env.actions[self.policy[tuple(state)]]

    def choose_action(self, state):
        ''' Uses the policy table to look up the best action in a given state
        args:
            state: The state being queried
        '''
        return self.policy[tuple(state)]

    def solve(self, mode='val', delta=1e-3, render_on=True, n_evals=1):
        ''' Solve the gridworld using dynamic programming
        kwargs:
            mode: 'val' for value iteration, 'pol' for policy iteration
            env: The environment to solve
            delta: Value iteration is stopped when the change from one iteration to the next is less than delta
            n_evals: The number of steps between policy evaluation and improvement (always set to 1 for value iteration)
        '''

        ## Value iteration does not have an embedded loop so we only do it once
        if mode == 'val':
            n_evals = 1

        ## Perform the iteration
        self.render() ## For showing the first policy and evaluation function
        if PAUSE: input('')

        for idx in count():

            ## Store the previous values to compare in the iteration
            v_old = self.values.copy()

            for i in range(n_evals):

                ## Iterate through all states
                for state in self.env.all_states:

                    ## Do not update the values of the end states!
                    if not myUT.isin(state, self.env.end_states):

                        ## See what actions are available to us, value iteration looks at all possible, policy iteration looks at current policy
                        pot_acts = [self.policy[tuple(state)]] if mode == 'pol' else np.arange(len(self.env.actions))

                        ## Find the values of each of those actions
                        poss_vals = []
                        for act in pot_acts:
                            next_state, reward, done = self.env.step(act, query_state=state)
                            next_value = self.values[tuple(next_state)]
                            poss_vals.append(reward + (not done) * self.gamma * next_value)

                        ## Update the state values based on the best action
                        self.values[tuple(state)] = np.max(poss_vals)

                    ## Update the value of coming into that state (can be thought of as the action value of the previous state)
                    ## This is what we base our policy on! Not just the value function!
                    self.inc_values[tuple(state)] = self.env.r_grid[tuple(state)] + self.gamma * self.values[tuple(state)]

                    ## For value iteration we improve the policy straight away
                    if mode == 'val':
                        self.update_policy(state)

                    if VIS_STATE_UPDATE:
                        if PAUSE: input('')
                        self.render(state) ## For showing the box move from state to state

                self.render() ## For showing the value function iterate and smoothen

            ## For policy iteration we improve the policy once it is fully evaluated
            if mode == 'pol':
                if PAUSE: input('')
                self.update_policy()
                self.render() ## For showing the policy arrows rotate
                if PAUSE: input('')

            ## Calculate the change in the values due to the iteration and break if smaller than the threshold
            if np.max(np.abs(v_old-self.values)) < delta:
                break

    def learn(self, *args):
        ''' A placeholder function to allow for the same interface as QAgent
        '''
        pass

    def render(self, state=None):
        ''' Visualise the value and policy grids as it is updated through the iteration methods.
        The first time the render method is called we must setup the figure, every subsequent call just updates.
        If the states argument a yellow box is drawn around it
        '''

        ## The fist call to render
        if not hasattr(self, 'fig'):
            plt.ion()

            ## Create value/policy figure
            self.fig, self.ax = plt.subplots(1, 1, figsize = (5,5))

            ## Create the values as a grid and the policy as a quiver
            self.val_grid = self.ax.imshow(self.inc_values.T, origin='lower', vmin=-10, vmax=10, cmap=rtg)
            self.pol_quiver = self.ax.quiver( *self.env.all_states.T,
                                                  self.polviz[:,:,0], self.polviz[:,:,1],
                                                  pivot='mid' )

            ## Create the highlighted box list to append and pop
            rec_dict = {'width':1, 'height':1, 'fc':'none', 'color':'blue', 'linewidth':2}
            self.box = Rectangle((-0.5, -0.5), **rec_dict)
            self.ax.add_patch(self.box)

            self.fig.tight_layout()
            self.ax.axes.xaxis.set_visible(False)
            self.ax.axes.yaxis.set_visible(False)

            ## Flush the canvas
            self.fig.canvas.flush_events()

        ## Update the data and the colours in the value grid
        self.val_grid.set_data(self.inc_values.T)
        self.val_grid.set_clim(np.min(self.inc_values), np.max(self.inc_values))

        ## Update the policty quiver
        self.pol_quiver.set_UVC(self.polviz[:,:,0], self.polviz[:,:,1])

        ## Draw the boxes
        if state is not None:
            self.box.set_xy(tuple(state-0.5))

        ## Flush the canvas
        self.fig.canvas.flush_events()


## Global Variables for visualisation steps
VIS_STATE_UPDATE = False
PAUSE = False

def main():

    ## Setup the environment
    env = GridWorldEnv(width=10, height=10,
                       rnd_strt=True,
                       stp_rwrd=-1,
                       chs_rwrd=50, n_chs=1,
                       cat_rwrd=-10, n_cat=20, cat_kill=True)
    env.render()

    ## Setup the mouse as a dynamic programming agent
    # mouse = DPAgent(env, gamma=0.99)
    # mouse.solve(delta = 1e-3, mode='val', n_evals=2)

    ## Setup the mouse as a Q-learning agent
    mouse = QAgent(env, gamma=0.99, lr=5e-1, eps=1.0, eps_min=1e-1, eps_dec=1e1)

    ## Iterate through episodes
    for ep in count():

        ## Reset the environment and get the starting state
        state = env.reset()

        ## Iterate through time
        for t in count():

            ## Visualise the environment
            env.render()

            ## Choose the action
            action = mouse.choose_action(state)

            ## The environment evolves wrt chosen action
            next_state, reward, done = env.step(action)

            ## Have the agent learn based off this transition
            mouse.learn(state, action, reward, next_state, done)

            ## Replacing the state
            state = next_state

            ## If we receive the signal, end the episode
            if done:
                break


if __name__ == '__main__':
    main()
