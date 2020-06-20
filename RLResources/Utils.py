import math
import numpy as np
import matplotlib.pyplot as plt
import torch as T
from collections import deque

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

class score_plot(object):
    def __init__(self, title = "", mvavg=20):
        self.mvavg = mvavg

        self.fig = plt.figure( figsize = (5,5) )
        self.ax  = self.fig.add_subplot(111)
        self.fig.suptitle(title)

        self.all_scores = deque(maxlen=10000)
        self.avg_scores = deque(maxlen=10000)

        self.score_line, = self.ax.plot( self.all_scores, "r." )
        self.avgs_line,  = self.ax.plot( self.avg_scores, "-k" )

    def update(self, ep_score):
        self.all_scores.append(ep_score)
        self.avg_scores = running_mean( self.all_scores, self.mvavg )

        self.score_line.set_data( np.arange(len(self.all_scores)), list(self.all_scores) )
        self.avgs_line.set_data( np.arange(len(self.avg_scores))+self.mvavg-1, list(self.avg_scores) )

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class dist_plot(object):
    def __init__(self, vmin, vmax, atoms):
        self.sups = np.linspace(vmin, vmax, atoms)

        self.fig = plt.figure( figsize = (5,5) )
        self.ax  = self.fig.add_subplot(111)

        self.ax.set_ylim([0,0.6])

        nulls = np.zeros( len(self.sups) )

        self.dist_line, = self.ax.plot( self.sups, nulls, "k-" )

    def update(self, dist):
        self.dist_line.set_data( self.sups, dist )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class value_plot(object):
    def __init__(self):

        self.fig = plt.figure( figsize = (5,5) )
        self.ax  = self.fig.add_subplot(111)

        self.values = deque(maxlen=1000)
        self.value_line,  = self.ax.plot( self.avg_scores, "-k" )

    def update(self, val):

        self.values.append(val)

        self.score_line.set_data( np.arange(len(self.values)), list(self.values) )

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()













