import matplotlib.pyplot as plt

def create_gridworld_axis(shape):
    '''
    Returns the figure and axis object which removes all ticklabels, and plots the gridlines.
    Is used throught the gridworld visualisation methods.
    args:
        shape: The size of the gridworld (width, height)
    '''

    fig, ax = plt.subplots(1, 1, figsize = (5,5))
    ax.set_aspect('equal', adjustable='box')

    ## Matching the dimensions of the plot to the gridworld
    ax.set_xlim([0, shape[0]])
    ax.set_ylim([0, shape[1]])

    ## Setting up the axis without any ticks or labels
    fig.tight_layout()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ## Drawing in the gridlines to indicate the different tiles/states
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True)
