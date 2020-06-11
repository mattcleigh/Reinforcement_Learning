import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
from matplotlib.animation import FuncAnimation
from scipy.special import softmax


class Bandit:

    ## We define the bandit by how many arms it has
    ## Our initial estimates of the values
    ## The epsilon for e-greedy algorithm

    def __init__(self, name = "Name", k_arm = 10, epsilon = 0, initial = 0, step_size = 0,
                 do_Bayesian = False, prior = 3, do_UCB = False, do_Gradient = False, alpha = 0.25):

        self.name = name
        self.k = k_arm
        self.epsilon = epsilon
        self.initial = initial
        self.indicies = np.arange(self.k)
        self.step_size = step_size

        self.do_Bayesian = do_Bayesian
        self.prior = prior

        self.do_UCB = do_UCB

        self.do_Gradient = do_Gradient
        self.alpha = alpha


    def reset(self):

        ## The average real reward for each arm
        self.q_true = np.random.randn(self.k)

        ## The estimation for each reward
        self.q_estimated = np.zeros(self.k) + self.initial

        ## The running squared of each reward
        self.q_squared = np.zeros(self.k)

        ## The uncertainties on the rewards
        self.q_uncertainties = np.zeros(self.k) + self.prior

        ## The number of times each action was recorded
        self.a_count = np.zeros(self.k)

        ## The preference of an action based on the gradient method
        self.preferences = np.zeros(self.k)

        ## The number of iterations
        self.time = 0

        ## The total average of all rewards
        self.q_average = 0

        ## Did it make the correct choice at a particular iteration
        self.accuracy = []

        ## The last action made by the agent
        self.last_action = 0

        ## The real best action
        self.best_true = np.argmax(self.q_true)


    def chose_action(self):

        ## If we are doing the gradient methods
        if self.do_Gradient:
            return np.random.choice( self.indicies, p=softmax(self.preferences) )

        ## If we are doing the interval methods
        if self.do_Bayesian or self.do_UCB:
            return np.argmax( self.q_estimated + self.q_uncertainties )

        ## We select the best action based on the e-greedy algorithm
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indicies)

        ## The greedy choice
        return np.argmax(self.q_estimated)

    def iterate(self):

        ## Decide on an action
        action = self.chose_action()
        self.last_action = action
        self.accuracy.append( action == self.best_true )

        ## Generate the reward given based on mean and noise
        reward = self.q_true[action] + np.random.randn()

        ## Update the number of recorded events
        self.time += 1
        self.a_count[action] += 1

        ## Update our averages using the step size
        step = 1 / self.a_count[action] if self.step_size == 0 else self.step_size
        self.q_estimated[action] += step * (reward - self.q_estimated[action])

        ## For the bayesian methods we need to update the uncertainties
        if self.do_Bayesian or self.epsilon == 1.0:
            self.q_squared[action] += reward**2
            sigmas = np.sqrt(( self.q_squared - self.a_count * self.q_estimated**2 ) / (self.a_count**1.5 - 1 + 1e-9 ))
            self.q_uncertainties = np.where( self.a_count > 5, sigmas, self.prior )

        ## For the UCB method we need to update the uncertainties
        if self.do_UCB:
            upper = self.q_estimated + np.sqrt( np.log(self.time+1) / (self.a_count + 1e-9) )
            self.q_uncertainties = np.where( self.a_count == 0, 10, upper )

        ## For the gradient methods, we need to update our preferences
        if self.do_Gradient:
            self.q_average += (reward - self.q_average) / self.time
            pref_change = self.alpha * (reward - self.q_average)
            soft = softmax( self.preferences )
            self.preferences += np.where( self.indicies==action, pref_change*(1-soft), -pref_change*(soft) )



## Create the bandit
def test_bed(bandit, runs, max_t):

    best_action_counts = [ [] for r in range(runs) ]

    for i in range(runs):
        bandit.reset()

        while bandit.time < max_t:
            bandit.iterate()

        best_action_counts[i] = bandit.accuracy

        print( "Finished Run - {}\r".format( i ), end="" )
        sys.stdout.flush()
    print( "Finished Run - {}\r".format( runs ) )

    return np.mean(best_action_counts, axis = 0)

def create_accuracy_plots( runs, max_t, ban_list ):
    fig = plt.figure()
    ax = fig.add_subplot( 1, 1, 1 )
    ax.set_xlabel("Time")
    ax.set_ylabel("Best Accuracy")

    ## Plot the accuracy for each bandit type
    for ban in ban_list:
        ban_accuracy = test_bed( ban, runs, max_t )
        ax.plot( ban_accuracy, label = ban.name )

    plt.legend()
    plt.show()


def animation_loop(it, bandit, max_t, estimation, updated, upper_bound, lower_bound, count ):

    bandit.iterate()
    estimation.set_data( bandit.indicies, bandit.q_estimated )
    updated.set_data( bandit.last_action, 0 )

    if bandit.do_Bayesian or bandit.epsilon == 1.0 or bandit.do_UCB:
        upper_bound.set_data( bandit.indicies, bandit.q_estimated + bandit.q_uncertainties )
    if bandit.do_Bayesian or bandit.epsilon == 1.0:
        lower_bound.set_data( bandit.indicies, bandit.q_estimated - bandit.q_uncertainties )

    hist_x = np.concatenate(( [-1], bandit.indicies, [len(bandit.indicies)] ))
    hist_y = np.concatenate(( [-4], 4*bandit.a_count/bandit.time-4, [-4] ))

    count.set_data( hist_x, hist_y )

    return estimation, updated, upper_bound, lower_bound, count


def animate_banit_learning( bandit, max_t ):

    ## Setting up the figure and the axis labels
    fig = plt.figure( figsize = (15, 10) )
    ax = fig.add_subplot( 1, 1, 1 )
    ax.set_xlabel("Action Slot")
    ax.set_ylabel("Reward")
    ax.set_xlim([-1, bandit.k])
    ax.set_ylim([-4, 4])

    ## Initialising the bandit
    bandit.reset()

    ## Plotting the truth values
    ax.errorbar( bandit.indicies, bandit.q_true, yerr = 1, fmt="rs", capsize=10, label = "Real Value")
    ax.axvspan( bandit.best_true-0.5, bandit.best_true+0.5, facecolor='green', alpha=0.5)

    ## Plotting the estimates which get updated with time
    estimation,  = ax.plot( bandit.q_estimated, "-bo", markersize=8, label = "Estimated Value" )
    updated,     = ax.plot( 0, "-kx", markersize=10 )
    upper_bound, = ax.plot( [], [], "-b", markersize=8 )
    lower_bound, = ax.plot( [], [], "-b", markersize=8 )
    count,       = ax.step( [], [], "-k", where = 'mid', )

    params = [bandit, max_t, estimation, updated, upper_bound, lower_bound, count]

    plt.legend()
    plt.draw()

    animation = FuncAnimation(fig, animation_loop, blit=True, fargs = params, interval = 0 )

    plt.show()



def main():

    ###### Starting the simulation
    runs  = 100
    max_t = 1000
    k_arm = 50

    bandit_list = [
                        Bandit(name = "E-greedy",       k_arm = k_arm, epsilon = 0.1, initial = 0),
                        Bandit(name = "Optimistic",     k_arm = k_arm, initial = 4, step_size = 0.05),
                        Bandit(name = "UCB",            k_arm = k_arm, do_UCB = True ),
                        Bandit(name = "Gradient",       k_arm = k_arm, do_Gradient = True, alpha = 0.1 ),
                        Bandit(name = "Bayesian",       k_arm = k_arm, do_Bayesian = True ),
                  ]

    # create_accuracy_plots( runs, max_t, bandit_list )
    animate_banit_learning( bandit_list[4], max_t )



if __name__ == '__main__':
    main()






























