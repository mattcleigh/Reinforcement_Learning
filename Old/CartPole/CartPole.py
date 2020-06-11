import time
import math
import matplotlib.pyplot as plt
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import Environment as myenv
import Learning_Systems as lsys

def main():


    ## The vairbales required for training
    mem_capacity    = 10000                 ## total number of memories to store
    batch_size      = 512                   ## number of memories to sample
    gamma           = 1.0                  ## future rewards discount (must be less than 1)
    epsilon         = ( 1.0, 0.01, 1000 )   ## start, end, and decay rate of epsilon
    target_update   = 10                    ## episode interval between target updates

    ## The device used by the system
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## The two networks are initialised
    policy_net = lsys.SimpleNet().to(device)
    target_net = lsys.SimpleNet().to(device)
    # policy_net.load_state_dict(torch.load("Balance_Network"))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    # policy_net.eval()

    ## Other objects are initialised
    optimizer = optim.Adam( policy_net.parameters(), lr = 0.001 )
    memory    = lsys.ReplayMemory( mem_capacity )
    loss_fn   = nn.SmoothL1Loss( reduction='none' )

    ## For plotting the updates
    plt.ion()
    fig = plt.figure( figsize = (5,5) )
    ax  = fig.add_subplot(111)
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 500])
    scores  = []
    mv_avg  = []
    score_line, = ax.plot( scores, "-g", markersize=10 )
    avg_line,   = ax.plot( mv_avg, "--r", markersize=10 )

    ## Initialise the testing environment and get the first state
    env = myenv.CartPoleEnv()

    all_time = 0
    for ep in count():
        ep_time = 0

        start_state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        state       = torch.tensor( env.state, device=device, dtype=torch.float32 ).unsqueeze(0)
        env.reset(*start_state)

        for t in count():
            env.render()

            ## We reduce eps based on the timestep
            eps = epsilon[1]+(epsilon[0]-epsilon[1])*math.exp(-all_time/epsilon[2])

            ## Choose an action
            policy_net.eval()
            action = lsys.e_greedy_action( policy_net, state, eps, device )

            ## Apply the action to observe new state and new reward
            next_state, reward, failed = env.step( action.item() )

            ## Convert those to tensors
            reward     = torch.tensor( reward, device=device, dtype=torch.float32).view(1,1)
            next_state = torch.tensor( next_state, device = device, dtype=torch.float32 ).unsqueeze(0) if not failed else None

            ## Store the transition in memory
            memory.push( state, action, next_state, reward )
            last_trans = lsys.Transition( state, action, next_state, reward )

            ## Move to the next state
            state = next_state

            ## Perform and optimisation step on the target network
            policy_net.train()
            lsys.train( policy_net, target_net, memory, batch_size, gamma, loss_fn, optimizer, last_trans, device )

            all_time += 1

            if failed:
                break

        ## We update the target network every so often
        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if ep % 100 == 0:
            print("Saving")
            torch.save( target_net.state_dict(), "Balance_Network" )

        ## Plotting and printing the results
        if len(scores)>=100: av = np.average( scores[-100:] )
        else: av = 0

        mv_avg.append(av)
        scores.append(t)

        if len(scores)>500: scores.pop(0)
        if len(mv_avg)>500: mv_avg.pop(0)

        score_line.set_data( np.arange(len(scores)), scores )
        avg_line.set_data(   np.arange(len(mv_avg)), mv_avg )
        fig.canvas.draw()
        fig.canvas.flush_events()
        print("Episode {:5}, score = {:5}, eps = {:6.2f}".format(ep, t, eps))







if __name__ == '__main__':
    main()