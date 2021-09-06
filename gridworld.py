import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

from itertools import count
from matplotlib.ticker import MultipleLocator

import Resources.utils as myUT

class GridWorldEnv:
    '''
    This is a simple grid based reinforcement learning environment, primarily for demonstration perposes.
    The environment
        keeps track of the current state
        updates the current state by receiving actions from an agent
        dishes out rewards based on new states
    We can also use the environment render function to allow us to observe the episode
    kwargs:
        width: The number of grid squares in the x direction
        height: The number of grid squares in the y direction
        rnd_strt: If True, the current state at the begining of the episode is chosen randomly
                  If False, the current state will always be in the top left-hand corner
        stp_rwrd: The reward given for taking any action (setting to small negative gives urgency)
        chs_rwrd: The reward given for finding the cheese (also a winning state which ends the episode)
        cat_rwrd: The reward/punishment given for running into a cat (meow)
        cat_kill: True if running into cat ends the episode
                  If False the agent will 'bounce off' while still incurring the penalty
    '''
    def __init__(self, width=8, height=8,
                       rnd_strt=False,
                       stp_rwrd=0,
                       chs_rwrd=10, n_chs=1,
                       cat_rwrd=-10, n_cat=10, cat_kill=True):

        ## Define the possible actions for the gridworld, which is just the 4 cardinal directions
        self.actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])

        ## Save the dimensions of the gridworld (used for checking out of bounds)
        self.shape = np.array([width, height])

        ## Save some of the configurations which will be used
        self.stp_rwrd = stp_rwrd
        self.rnd_strt = rnd_strt
        self.cat_kill = cat_kill

        ## Place the winning (cheese) and loosing (cat) states into the grid (cant be on the first corner)
        end_states = rd.choice(np.arange(1, np.prod(self.shape)), n_chs+n_cat, replace=False)
        self.end_states = np.array(np.unravel_index(end_states, tuple(self.shape))).T
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
        '''
        This resets the location of the mouse for the start of a new episode
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
        '''
        This function advances the environment by taking in an action signal sent by the appropriate agent
        args:
            action: This is a np.array of two intergers for which direction to move on the gird
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
        '''
        This function draws the gridworld and its inhabitants.
        We use matplotlib to render the world as it is basic enough, but this can always be changed!
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

# class RLAgent

class DPAgent:
    '''
    This is an RL agent which can interact with a basic deterministic grid world.
    It picks certain actions based on its state value function which may still be
    be interpreted as an action value function.
    Exploitation vs exploitation balance is done through epsilon greey actions
    Calculation of the value function can be done in 1 of 3 ways
     - MC: Keeps a running history of the return and updates at the end of the episode
     - SARSA: Uses on policy using the on policy SARSA algorithm
     - QLRN: Uses the off policy double q-learning algorithm
     Alternatively we can use value iteration methods to completely solve the gridworld
    '''
    def __init__(self, env, gamma=0.99):
        '''
        args:
            env: The environment this agent will act on. This is required as dynamic programming requires
                 access a copy of the state transition matrix (the step() function) to solve for the values
        kwargs:
            gamma: The discount factor for future rewards
        '''

        ## Store the gamma factor
        self.gamma = gamma

        ## Save a copy of the environment and initialise a state value table
        self.env = env
        self.values = np.zeros(env.shape)

        ## The value of coming into a state (good for visualisation)
        ## Essentially the action values related to state values by Bellman
        self.inc_values = np.zeros(env.shape)

    def choose_action(self, state):

        ## Calculate the action values for each possible action using the next states they lead to
        next_states = np.array([self.env.step(act, query_state=state)[0] for act in self.env.actions])
        returns = self.inc_values[tuple(next_states.T)]

        ## Choose the best action that maximises this return
        best_act = self.env.actions[np.argmax(returns)]

        return best_act


    def solve(self, delta = 1e-3, render_on=True):
        '''
        Solve the gridworld using dynamic programming and value iteration
        args:
            env: The environment to solve
            delta: Value iteration is stopped when the change from one iteration to the next is less than delta
        '''

        ## Get a list of every possible state in the gridworld
        all_states = np.array(np.unravel_index(np.arange(np.prod(self.values.shape)), self.values.shape)).T

        ## Perform the iteration
        for idx in count():

            ## Store the previous values to compare in the iteration
            v_old = self.values.copy()

            ## Iterate through all states
            for state in all_states:

                ## Update the value of coming into that state (can be thought of as the action value of the previous state)
                ## This is what we base our policy on! Not just the value function!
                self.inc_values[tuple(state)] = self.env.r_grid[tuple(state)] + self.gamma * self.values[tuple(state)]

                ## The value of an end state is always zero! The episode is over!
                if myUT.isin(state, self.env.end_states):
                    continue

                ## Find the values of all action in this state
                poss_vals = []
                for act in self.env.actions:
                    next_state, reward, done = self.env.step(act, query_state=state)
                    next_value = self.values[tuple(next_state)]
                    poss_vals.append(reward + (not done) * self.gamma * next_value)

                ## Update the state values and the incoming (action) values based on the best action
                self.values[tuple(state)] = np.max(poss_vals)

            ## Render the latest update
            self.render()

            ## Calculate the change in the values due to the iteration and break if smaller than the threshold
            if np.max(np.abs(v_old-self.values)) < delta:
                break

    def render(self):
        '''
        This is used to visualise the values and policy grids as it is updated through the iteration methods.
        The first time the render method is called we must setup the figure, every subsequent call just updates.
        '''

        ## The fist call to render
        if not hasattr(self, 'val_fig'):
            plt.ion()

            ## Create the value grid
            self.val_fig, self.val_ax = plt.subplots(1, 1, figsize = (5,5))

            self.val_fig.tight_layout()
            self.val_ax.axes.xaxis.set_visible(False)
            self.val_ax.axes.yaxis.set_visible(False)

            ## Good colours for the value grid can be approximated from the reward grid and the geometric sequence formula
            stp_r = self.env.stp_rwrd * (1-self.gamma ** self.env.shape[0]) / (1-self.gamma)
            max_r = max(np.max(self.env.r_grid), 0)
            min_r = np.min(self.env.r_grid) + stp_r

            ## Create the grid
            self.val_grid = self.val_ax.imshow(self.inc_values.T, origin='lower', cmap='inferno')

            ## Flush the canvas
            self.val_fig.canvas.flush_events()

        ## Update the data and the colours in the value grid
        self.val_grid.set_data(self.inc_values.T)
        self.val_grid.set_clim(np.min(self.inc_values), np.max(self.inc_values))

        ## Flush the canvas
        self.val_fig.canvas.flush_events()


def main():

    render_on = True

    ## Setup the environment
    env = GridWorldEnv(width=20, height=20,
                       rnd_strt=True,
                       stp_rwrd=-10,
                       chs_rwrd=10, n_chs=1,
                       cat_rwrd=-10, n_cat=1, cat_kill=True)

    ## Setup the mouse as a dynamic programming agent
    mouse = DPAgent(env, gamma=0.99)
    mouse.solve(delta = 1e-1, render_on=render_on) ## Use value iteration to solve the gridworld perfectly

    ## Iterate through episodes
    for ep in count():

        ## Reset the environment and get the starting state
        state = env.reset()

        ## Iterate through time
        for t in count():

            ## Visualise the environment if we want
            if render_on:
                env.render()

            ## Choose the action
            action = mouse.choose_action(state)

            ## The environment evolves wrt chosen action
            next_state, reward, done = env.step(action)

            ## Replacing the state
            state = next_state

            ## If we receive the signal, end the episode
            if done:
                break


if __name__ == '__main__':
    main()
