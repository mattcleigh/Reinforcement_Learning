import sys
import textwrap
import itertools
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__( self, grid_size = 5, gamma = 0.9 ):
        self.gamma = gamma

        self.grid_size = grid_size

        self.state_values  = np.zeros( [grid_size, grid_size] )
        self.actions       = np.array( [ [0,1], [0,-1], [-1,0], [1,0] ] )
        self.terminals     = [ (0, grid_size-1), (grid_size-1, 0) ]

    def list_successor_states(self, x, y):
        action = self.actions[0]
        state_new = tuple ( np.clip( [x, y] + action, 0, self.grid_size-1 ) )
        print (tuple(state_new))
        print( self.state_values[state_new] )
        print( self.state_values[2,2] )

    def iterative_policy_evaluation(self, thres = 1e-5):
        delta = thres + 1
        while delta > thres:
            delta = 0

            for x in range(self.grid_size):
                for y in range(self.grid_size):

                    if (x,y) in self.terminals:
                        continue


                    ## Save the old state (for difference measurements)
                    v_old = self.state_values[x,y]

                    ## Calculate the new value using the bellman equation
                    v_new = 0
                    for a in self.actions:
                        state_new = tuple( np.clip( [x, y] + a, 0, self.grid_size-1 ) )
                        v_new += ( -1 + self.gamma * self.state_values[state_new] ) / 4
                    self.state_values[x,y] = v_new
                    delta = max( delta, np.abs( v_new - v_old ) )

            self.render_update()

    def render_start(self):

        ## Making it such that the plot can be animated
        plt.ion()

        ## Creating the figure and axis as class attributes
        self.fig = plt.figure( figsize = (6,6) )
        self.ax  = self.fig.add_subplot(111)

        ## Creating the grid
        self.ax.set_xlim([0, self.grid_size])
        self.ax.set_ylim([0, self.grid_size])
        self.ax.set_xticks(np.arange(0, self.grid_size, 1))
        self.ax.set_yticks(np.arange(0, self.grid_size, 1))
        self.ax.xaxis.set_ticklabels([])
        self.ax.yaxis.set_ticklabels([])
        for tic in self.ax.xaxis.get_major_ticks() + self.ax.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False)
        self.ax.grid(True)

        ## Filling the grid with the state values
        self.text_list = np.empty( [self.grid_size, self.grid_size], dtype=object )
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.text_list[x,y] = self.ax.text( x+0.5, y+0.5, "{:.2f}".format(self.state_values[x,y]), ha="center", va="center", size=16 )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render_update(self):

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.text_list[x,y].set_text("{:.2f}".format(self.state_values[x,y]))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

my_world = GridWorld( grid_size = 5, gamma = 1)
my_world.render_start()
my_world.iterative_policy_evaluation( thres = 0 )

plt.ioff()
plt.show()

















