
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
    mem_capacity    = 500000                ## total number of memories to store
    batch_size      = 512                   ## number of memories to sample
    gamma           = 0.999                 ## future rewards discount (must be less than 1)
    epsilon         = [1.0, 0.00, 200 ]    ## start, end, and decay rate of epsilon
    load_previous   = True

    ## The device used by the system
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## The two networks are initialised
    policy_net = lsys.SimpleNet().to(device)
    target_net = lsys.SimpleNet().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    ## Other objects are initialised
    optimiser = optim.Adam( policy_net.parameters(), lr = 0.0001 )
    memory    = lsys.ReplayMemory( mem_capacity )
    loss_fn   = nn.SmoothL1Loss( reduction='none' )

    ## Loading the previous state of the network and optimiser
    if load_previous:
        policy_net.load_state_dict( torch.load("Policy_Network") )
        target_net.load_state_dict( torch.load("Target_Network") )
        optimiser.load_state_dict(  torch.load("Optimiser_State") )
        epsilon[0] = epsilon[1]

    ## Create the learning environment
    env = myenv.DrivingEnv()

    all_time = 0
    for ep in count():

        state = env.reset()
        state = torch.tensor( state, device=device, dtype=torch.float32 ).unsqueeze(0)
        ep_score = 0

        ## Reduce the exploration rate based on episode
        eps = epsilon[1]+(epsilon[0]-epsilon[1])*math.exp(-ep/epsilon[2])

        for it in count():
            env.render()

            ## Chose an action (a number from 0, to 11 )
            action = lsys.e_greedy_action( policy_net, state, eps, device )

            ## Apply the action to observe a new state and reward
            next_state, reward, failed = env.step( action.item() )

            ## Convert these into torch tensors
            reward     = torch.tensor( reward, device=device, dtype=torch.float32).view(1,1)
            next_state = torch.tensor( next_state, device=device, dtype=torch.float32 ).unsqueeze(0) if not failed else None

            ## Store the transition in memory
            memory.push( state, action, next_state, reward )
            last_trans = lsys.Transition( state, action, next_state, reward )

            ## Move to the next state
            state = next_state

            ## Perform an optimisation step on the target network
            lsys.train( policy_net, target_net, memory, batch_size, gamma, loss_fn, optimiser, last_trans, device )

            ## Increase the ticker and update the score
            all_time += 1
            ep_score += reward.item()

            ## We save good scoring networks
            if ep_score == 30:
                print("Saving")
                torch.save( policy_net.state_dict(), "Good_Policy_Network" )
                torch.save( target_net.state_dict(), "Good_Target_Network" )
                torch.save( optimiser.state_dict(),  "Good_Optimiser_State" )

            ## We exit the loop when the fail state is reached
            if failed:
                break

        if ep % 50 == 0:
            print("Saving")
            torch.save( policy_net.state_dict(), "Policy_Network" )
            torch.save( target_net.state_dict(), "Target_Network" )
            torch.save( optimiser.state_dict(),  "Optimiser_State" )


        print( "Episode {}: Reward = {:4.2f}, Epsilon = {:4.2f}, Time = {}".format( ep, ep_score, eps, all_time ) )



if __name__ == '__main__':
    main()
















