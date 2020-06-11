import sys
import textwrap
import itertools
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__( self, width, height, start, end, render_on = True, verbose = False, numbers = True ):

        ## We define the dimensions of the gridworld and the locations of the start and end states
        self.width  = width
        self.height = height
        self.start_state = start
        self.end_state   = end

        ## The list of all possible actions and the action values for each state
        self.actions_names = [ "up", "down", "left", "right", "upleft", "upright", "downleft", "downright" ]
        self.actions  = np.arange(len(self.actions_names))
        self.q_values = np.zeros( (self.width, self.height, len(self.actions)) )

        ## Whether we want full terminal outputs or graphical displays
        self.verbose = verbose
        self.render_on = render_on
        self.numbers = numbers

    def initialise(self, lrn_rate, exp_rate):

        ## The learning and exploration rates for the agent and sarsa method
        self.lrn_rate = lrn_rate
        self.exp_rate = exp_rate

        self.q_values = np.zeros( (self.width, self.height, len(self.actions)) )

        ## For rendering we need a history of the agent
        self.agent_history = [ self.start_state, self.start_state ]


    ## Initialising the canvas for rendering
    def render_start(self):
        plt.ion()

        ## Creating the figure and axis as class attributes
        self.fig = plt.figure( figsize = (10,10) )
        self.ax  = self.fig.add_subplot(111)

        ## Creating the grid
        self.ax.set_xlim([0, self.width])
        self.ax.set_ylim([0, self.height])
        self.ax.set_xticks(np.arange(0, self.width, 1))
        self.ax.set_yticks(np.arange(0, self.height, 1))
        self.ax.xaxis.set_ticklabels([])
        self.ax.yaxis.set_ticklabels([])
        for tic in self.ax.xaxis.get_major_ticks() + self.ax.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False)

        ## Filling the grid with the state values
        if self.height * self.width < 200 and self.numbers:
            self.ax.grid(True)
            self.text_list = np.empty( [self.width, self.height], dtype=object )
            for x in range(self.width):
                for y in range(self.height):
                    val = max(self.q_values[x,y])
                    self.text_list[x,y] = self.ax.text( x+0.5, y+0.5, "{:.2f}".format(val), ha="center", va="center", size=15 )

        ## Placing the icons of the end state, the agent, and its trail
        self.end_icon,   = self.ax.plot( self.end_state[0]+0.5,   self.end_state[1]+0.5,   "ro", markersize=10 )
        self.agent_icon, = self.ax.plot( self.start_state[0]+0.5, self.start_state[1]+0.5, "g<", markersize=10 )
        self.agent_path, = self.ax.plot( [], [], "-g" )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        for i in range(2):
            self.render_update( self.start_state )


    def render_update(self, current_state):

        ## First we append to the path which is only used for rendering
        self.agent_history.append( current_state )
        hist_arr = np.array ( self.agent_history )

        ## Then we update the numbers
        if self.height * self.width < 200 and self.numbers:
            last_state = self.agent_history[-2]
            val = max(self.q_values[last_state])
            self.text_list[last_state].set_text("{:.1f}".format(val))

        self.agent_icon.set_data( current_state[0] + 0.5, current_state[1] + 0.5 )
        self.agent_path.set_data( hist_arr[:,0]    + 0.5, hist_arr[:,1]    + 0.5 )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



    ## A function to chose the action based on the e-greedy method
    def get_action(self, current_state, force = None):

        if self.verbose:
            print("chosing action ...")

        ## First we check what kind of step this is, by checking specification
        if force is not None:
            type = force
        ## If not explicitly requested then we do the e-greedy test
        else:
            type = "explore" if np.random.random() < self.exp_rate else "greedy"

        ## We check if we want an explorative action
        if type == "explore":
            act = np.random.choice( self.actions )

            if self.verbose:
                print("   exploring ...")

        ## Otherwise we make a greedy action
        if type == "greedy":
            action_values = self.q_values[current_state]
            best_actions  = self.actions[ np.where( action_values == np.max(action_values) ) ]
            act = np.random.choice( best_actions )

            if self.verbose:
                print("   greedy ...")
                print("   {}".format(action_values) )

        ## The action is returned in the form descibing its idx
        if self.verbose:
            print("   {}!".format(self.actions_names[act]))
        return act

    ## Returning the next state given an action
    def get_next_state(self, current_state, action):

        ## It is easier to keep track of the actions by their name
        action_name = self.actions_names[action]

        if self.verbose:
            print("moving to new state ...")
            print("   old:    {}".format(current_state) )
            print("   action: {}".format(action_name)   )

        ## Using the name we can see the effect an action has on the coordinates
        if   action_name == "up":    effect = [  0,  1 ]
        elif action_name == "down":  effect = [  0, -1 ]
        elif action_name == "left":  effect = [ -1,  0 ]
        elif action_name == "right": effect = [  1,  0 ]
        elif action_name == "upleft":    effect = [ -1,  1 ]
        elif action_name == "upright":   effect = [  1,  1 ]
        elif action_name == "downleft":  effect = [ -1, -1 ]
        elif action_name == "downright": effect = [  1, -1 ]

        next_s = np.array(current_state) + effect

        ## Now we add in the wind
        wind = np.array([  0,  0 ])
        if 3 <= current_state[0] <= 8:
            wind += [  0,  1 ]
        if 6 <= current_state[0] <= 7:
            wind += [  0,  1 ]
        next_s = next_s + wind

        ## The state must adhere to the walls of the gridworld
        next_x = np.clip( next_s[0], 0, self.width-1)
        next_y = np.clip( next_s[1], 0, self.height-1)

        if self.verbose:
            print("   new:    {}".format( ( next_x, next_y ) ) )

        return ( next_x, next_y )

    ## The reward given state and action combinations
    def get_reward( self, current_state ):

        ## For SARSA terminal states must give zero reward
        if current_state == self.end_state:
            return 0
        ## Every other state privides a punishment to urge quickness
        return -1


    ## The function that begins the episodes on the gridworld
    def play(self, algorithm = "sarsa", max_episodes = 200, display_from = 1, test_from = 0):

        print("\n\n STARTING GRIDWORLD TRAINING SESSION\n\n")
        time_taken = []

        ## Begin the episode counter
        for ep in range(1, max_episodes+1):
            print("\nEpisode {}".format(ep) )

            ## We initialise the render method on a certain episode
            if ep == display_from and self.render_on:
                self.render_start()
                print("\n\nRender Initialising ...")
                input("Press ENTER to continue ...\n\n")

            if ep == test_from:
                self.exp_rate = 0

            ## Resetting the position of the agent
            state  = self.start_state
            action = self.get_action( state )

            ## Resetting all other necc values
            steps_this_ep = 0
            self.agent_history = [ state, state ]
            # self.exp_rate = ( 1 / ep )

            while True:
                if self.verbose:
                    input("")

                steps_this_ep += 1

                ## Calculating the reward and the next state-action pair
                reward      = self.get_reward( state )
                next_state  = self.get_next_state( state, action )

                if   algorithm == "sarsa":     next_action = self.get_action( next_state )
                elif algorithm == "qlearning": next_action = self.get_action( next_state, force = "greedy" )

                ## We render using the new positions
                if self.render_on and (ep>=display_from):
                    self.render_update(next_state)


                ## Applying the SARSA algorithm to update the previous state-action-value
                trpl   = ( state[0], state[1], action )
                n_trpl = ( next_state[0], next_state[1], next_action )
                self.q_values[trpl] += self.lrn_rate * ( reward + self.q_values[n_trpl] - self.q_values[trpl] )

                ## Updating our current states and actions
                state = next_state
                if   algorithm == "sarsa":      action = next_action
                elif algorithm == "qlearning":  action = self.get_action( state )

                ## If we are in a terminal state we must start a new episode:
                if state == self.end_state:
                    time_taken.append( steps_this_ep )
                    break

        return(time_taken)







def main():

    my_world = GridWorld( width = 10, height = 7, start = (0,3), end = (7,3), render_on = True, verbose = False, numbers = True )

    # my_world.initialise( lrn_rate = 0.5, exp_rate = 0.1 )
    # sarsa_time = my_world.play(algorithm = "sarsa", max_episodes = 2000, display_from = 1900, test_from = 1900)

    my_world.initialise( lrn_rate = 0.2, exp_rate = 0.5 )
    ql_time = my_world.play(algorithm = "qlearning", max_episodes = 2000, display_from = 1000, test_from = 1000)

    plt.ioff()
    plt.show()

    plt.close('all')
    plt.plot( sarsa_time )
    plt.plot( ql_time )
    # plt.yscale("log")
    plt.show()

if __name__ == "__main__":
    main()




























