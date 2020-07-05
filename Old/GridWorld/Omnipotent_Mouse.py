import sys
import textwrap
import itertools
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__( self, width = 8, height = 8, n_cats = 10,
                  lrn_rate = 0.1, expl_rate = 0.2, expl_decay = 0.99,
                  render_on = True, verbose = False, numbers = False ):

        self.width  = width
        self.height = height

        ## The learned values of each state
        self.state_values = np.zeros( (width, height) )

        ## The location of the terminal states
        possible_states   = np.array([ (x, y) for x in range(0, width) for y in range(0, height) ])
        chosen_end_states = np.random.choice( len(possible_states), size = n_cats+1, replace = None )
        self.end_states   = [ tuple(possible_states[i]) for i in chosen_end_states ]
        self.cat_states   = self.end_states[:-1]
        self.win_states   = [ self.end_states[-1] ]

        ## The properties of the mouse
        self.lrn_rate   = lrn_rate
        self.expl_rate  = expl_rate
        self.expl_decay = expl_decay
        self.mouse_actions       = [ 0,    1,      2,      3       ]
        self.mouse_action_names  = [ "up", "down", "left", "right" ]
        self.mouse_start_state   = ( 0, height-1 )
        self.mouse_current_state = self.mouse_start_state
        self.mouse_state_history = [ self.mouse_start_state ]
        self.mouse_state_path    = [ self.mouse_start_state ]

        self.render_on = render_on
        self.verbose = verbose
        self.numbers = numbers

        if render_on:
            self.render_start()


    def value_iteration(self, thres = 1e-6):

        delta = thres + 1
        idx = 1
        while delta >= thres:


            if self.verbose:
                input("     awaiting click     \n")

            delta = 0

            for x in reversed(range(self.width)):
                for y in range(self.height):


                    if (x,y) in self.end_states:
                        continue

                    v_old = self.state_values[x, y]

                    v_next = -1e6
                    for act in self.mouse_actions:
                        state_new  = self.next_state( (x, y), act )
                        value_new  = self.state_values[ state_new ]
                        reward_new = self.give_reward( state_new )
                        v_next = max( v_next, reward_new + self.lrn_rate * value_new )

                    self.state_values[x, y] = v_next
                    delta = max( delta, np.abs( v_old - v_next ) )

            if self.render_on and (self.height * self.width < 200):
                self.render_update()

            print( "value iteration: {}    delta = {:.5f}".format(idx, delta) )
            idx += 1

        for state in self.end_states:
            self.state_values[state] = self.give_reward(state)



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
                    self.text_list[x,y] = self.ax.text( x+0.5, y+0.5, "{:.2f}".format(self.state_values[x,y]), ha="center", va="center", size=15 )

        ## Placing the scatter of the cats, cheese and mouse
        cat_array = np.array( self.cat_states ) + 0.5
        self.cat_icon,     = self.ax.plot( cat_array[:,0], cat_array[:,1], "ro", markersize=10 )

        win_array = np.array( self.win_states ).ravel() + 0.5
        self.cheese_icon,  = self.ax.plot( win_array[0], win_array[1], "ys", markersize=10 )

        mouse_array = np.array( self.mouse_current_state ) + 0.5
        self.mouse_icon, = self.ax.plot( mouse_array[0], mouse_array[1], "g<", markersize=10 )

        start_array = np.array( self.mouse_start_state ) + 0.5
        self.start_icon, = self.ax.plot( start_array[0], start_array[1], "bo", markersize=10 )

        mouse_path = np.array( self.mouse_state_path ) + 0.5
        self.path_line, = self.ax.plot( mouse_path[:,0], mouse_path[:,1], "-g" )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        for i in range(2):
            self.render_update()


    def render_update(self):

        if self.height * self.width < 200 and self.numbers:
            for x in range(self.width):
                for y in range(self.height):
                    self.text_list[x,y].set_text("{:.1f}".format(self.state_values[x,y]))

        mouse_array = np.array( self.mouse_current_state ) + 0.5
        start_array = np.array( self.mouse_start_state ) + 0.5
        self.mouse_icon.set_data( mouse_array[0], mouse_array[1] )
        self.start_icon.set_data( start_array[0], start_array[1] )

        mouse_path = np.array( self.mouse_state_path + [ self.mouse_current_state ] ) + 0.5
        self.path_line.set_data( mouse_path[:,0], mouse_path[:,1] )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    ## Get the reward value of the current state
    def give_reward(self, state):

        if state in self.win_states:
            return 50
        if state in self.cat_states:
            return -1
        return 0

    ## Get the next state given an action
    def next_state(self, state, action):

        if   action == 0: a = [  0,  1 ]
        elif action == 1: a = [  0, -1 ]
        elif action == 2: a = [ -1,  0 ]
        elif action == 3: a = [  1,  0 ]

        next = np.array(state) + a
        next_x = np.clip( next[0], 0, self.width-1)
        next_y = np.clip( next[1], 0, self.height-1)

        return( next_x, next_y )

    def choose_action(self, force_explr = False ):

        # We limit our actions to the ones which wont make us backtrack
        viable_actions   = []
        viable_values    = []
        for act in self.mouse_actions:
            next_pos = self.next_state( self.mouse_current_state, act )
            if next_pos not in self.mouse_state_history:
                viable_actions.append( act )
                viable_values.append( self.state_values[ next_pos ] )

        ## If we can only backtrack then do so randomly
        if len( viable_values ) == 0:
            viable_actions = self.mouse_actions
            force_explr = True
            if self.verbose:
                print( " --> having to backtrack " )


        ## We may want to explore but we dont run into walls
        if np.random.random() < self.expl_rate or force_explr:
            best_actions = viable_actions
            type = 0
            if self.verbose:
                print( " --> exploratory" )

        ## For a normal greedy action
        else:
            best_idx = np.argwhere( viable_values == np.max(viable_values) ).ravel()
            best_actions = np.take(viable_actions, best_idx)
            type = 1
            if self.verbose:
                print( " --> exploitative" )


        a_idx = np.random.choice(best_actions)
        if self.verbose:
            print( " ----> {}".format( self.mouse_action_names[a_idx] ) )
        return self.mouse_actions[a_idx], type


    def reset(self, expl_start):

        if expl_start:
            self.mouse_start_state = ( np.random.randint(self.width), np.random.randint(self.height) )
            while self.mouse_start_state in self.end_states:
                self.mouse_start_state = ( np.random.randint(self.width), np.random.randint(self.height) )
        else:
            self.mouse_start_state   = ( 0, height-1 )
        self.mouse_current_state = self.mouse_start_state
        self.mouse_state_history = [ self.mouse_start_state ]
        self.mouse_state_path    = [ self.mouse_start_state ]



    def mc_play(self, max_rounds = 200, display_on = 0, test_mode = 100, expl_start = False):
        rnd = 0
        self.reset( expl_start )

        print( "Starting Test\n\n".format(rnd) )

        while rnd < max_rounds:


            if self.mouse_current_state in self.end_states:


                reward = self.give_reward( self.mouse_current_state )
                self.state_values[self.mouse_current_state] = reward

                rnd += 1
                print( "Finished Run - {}".format( rnd ) )
                if reward>0:
                    print("WIN!")
                else:
                    print("Lost!")

                for s in reversed(self.mouse_state_path):
                    reward = self.state_values[s] + self.lrn_rate * ( reward - self.state_values[s] )
                    self.state_values[s] = reward

                self.reset( expl_start )


            else:

                ## List of unique locations (to prevent backtracking)
                if self.mouse_current_state not in self.mouse_state_history:
                    self.mouse_state_history.append( self.mouse_current_state )

                ## Updating the path
                new = self.mouse_current_state
                new_loc = -1
                for e, (x,y) in enumerate(self.mouse_state_path):
                    if new == (x,y):
                        new_loc = e
                if new_loc==-1:
                    self.mouse_state_path.append( self.mouse_current_state )
                else:
                    self.mouse_state_path = self.mouse_state_path[:new_loc+1]
                    if self.verbose:
                        print(" --> loop detected, forcing exploratory move")

                action, type = self.choose_action()
                self.mouse_current_state = self.next_state( self.mouse_current_state, action )

            if rnd >= display_on and self.render_on:
                self.render_update()

            if rnd >= test_mode:
                self.expl_rate = 0
                expl_start = False

            if self.verbose:
                print()
                input("     awaiting click     \n")



    def td_play(self, max_rounds = 200, display_on = 0, test_mode = 100, expl_start = False):
        rnd = 0
        self.reset( expl_start )

        print( "Starting Test\n\n".format(rnd) )

        while rnd < max_rounds:

            ## If we are in an end state then we must reset
            if self.mouse_current_state in self.end_states:
                rnd += 1

                reward = self.give_reward( self.mouse_current_state )
                self.state_values[self.mouse_current_state] = reward

                self.reset( expl_start )

            else:

                ## First we chose a greedy action
                action, type = self.choose_action()

                ## Then we update the mouse position
                self.mouse_current_state = self.next_state( self.mouse_current_state, action )

                ## Then we update the history of the mouse positions
                if self.mouse_current_state not in self.mouse_state_history:
                    self.mouse_state_history.append( self.mouse_current_state )

                ## Then we update the path (since this is a markov process)
                new_loc = -1
                for t, (x,y) in enumerate(self.mouse_state_path):
                    if self.mouse_current_state == (x,y):
                        new_loc = t
                if new_loc == -1: self.mouse_state_path.append( self.mouse_current_state )
                else:             self.mouse_state_path = self.mouse_state_path[:new_loc+1]

                ## Now we calculate the reward received moving to this new state
                reward = self.give_reward( self.mouse_current_state )

                ## Now we apply the TD update equation to the previous state in the path
                ps = self.mouse_state_path[-2]
                self.state_values[ps] = self.state_values[ps] + self.lrn_rate * ( 0.9 * self.state_values[self.mouse_current_state] - self.state_values[ps] )

            if rnd >= display_on and self.render_on:
                self.render_update()

            if rnd >= test_mode:
                self.expl_rate = 0
                expl_start = False

            if self.verbose:
                print()
                input("     awaiting click     \n")


def main():

    my_world = GridWorld( width = 20, height = 20 , n_cats = 100,
                          lrn_rate = 0.9, expl_rate = 0.0, expl_decay = 1.0,
                          render_on = True, verbose = False, numbers = True)

    my_world.value_iteration()
    # my_world.lrn_rate = 0.2

    my_world.td_play( max_rounds = 1000, display_on = 0, test_mode = 900, expl_start = True)
    # my_world.mc_play( max_rounds = 1000, display_on = 0, test_mode = 900, expl_start = True)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
