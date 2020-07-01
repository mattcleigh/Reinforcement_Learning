import sys
home_env = '../../../Reinforcement_Learning/'
sys.path.append(home_env)

from Resources import MemoryMethods as mm
from Resources.Utils import score_plot
from Environments import Car_Env

import gym
import os
import time
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict

class Worker:
    """ This worker object contains its own environment and
        records its own experience.
        It is linked to the central agent.
    """
    def __init__(self, env_name, cen_agent, n_frames, gamma):

        if env_name=="car":
            self.env = Car_Env.MainEnv( rand_start = True )
        else:
            self.env = gym.make(env_name)

        self.state = self.env.reset()
        self.cen_agent = cen_agent
        self.n_frames = n_frames
        self.gamma = gamma
        self.ep_score = 0.0

    def fill_batch(self):
        states, actions, rewards, dones = [], [], [], []

        for _ in range(self.n_frames):
            action = self.cen_agent.choose_action(self.state)
            next_state, reward, done, info = self.env.step(action)

            states.append(self.state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            self.ep_score += reward
            self.state = next_state

            if done:
                self.state = self.env.reset()
                self.cen_agent.sp.update(self.ep_score)
                self.ep_score = 0.0

        ## Now that the batch is full we try calculate the n_step returns
        values = []

        ## The next value after our final action is 0 unless the episode continues
        next_value = 0
        if not dones[-1]:
            state_tensor = T.tensor( [states[-1]], device=self.cen_agent.actor_critic.device, dtype=T.float32 )
            next_value = self.cen_agent.actor_critic.get_value(state_tensor).item()

        ## From there we begin discounting and working backward reseting at each ep lim
        for i in reversed(range(self.n_frames)):
            if not dones[i]:
                next_value = rewards[i] + next_value * self.gamma
            else:
                next_value = rewards[i]
            values.append(next_value)

        values.reverse()

        ## Now we iterate through the new batch and store it to memory
        for s, a, v in zip(states, actions, values):
            self.cen_agent.store_transition( s, a, v )



class ActorCriticMLP(nn.Module):
    """ A simple and configurable multilayer perceptron.
        An actor-critic method usually includes one network each.
        However, feature extraction usually requires the same tools.
        Thus, they share the same base layer.
    """
    def __init__(self, name, chpt_dir,
                       input_dims, n_actions,
                       depth, width, activ ):
        super(ActorCriticMLP, self).__init__()

        ## Defining the network features
        self.name       = name
        self.chpt_dir   = chpt_dir
        self.chpt_file  = os.path.join(self.chpt_dir, self.name)
        self.input_dims = input_dims
        self.n_actions  = n_actions

        ## Defining the base (shared) layer structure
        layers = []
        for l_num in range(1, depth+1):
            inpt = input_dims[0] if l_num == 1 else width
            layers.append(( "base_lin_{}".format(l_num), nn.Linear(inpt, width) ))
            layers.append(( "base_act_{}".format(l_num), activ ))
        self.base_stream = nn.Sequential(OrderedDict(layers))

        ## Defining the actor network, returns the policy (softmax)
        self.actor_stream = nn.Sequential(OrderedDict([
            ( "actor_lin_out", nn.Linear(width, n_actions) ),
            ( "actor_act_out", nn.Softmax(dim=-1) ),
        ]))

        ## Defining the critic network, returns the state value function
        self.critic_stream = nn.Sequential(OrderedDict([
            ( "critic_lin_out", nn.Linear(width, 1) ),
        ]))

        ## Moving the network to the device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        shared_out = self.base_stream(state)
        policy = self.actor_stream(shared_out)
        value  = self.critic_stream(shared_out)
        return policy, value

    def get_value(self, state):
        shared_out = self.base_stream(state)
        value = self.critic_stream(shared_out)
        return value

    def get_policy(self, state):
        shared_out = self.base_stream(state)
        policy = self.actor_stream(shared_out)
        return policy

    def save_checkpoint(self, flag=""):
        print("... saving network checkpoint ..." )
        T.save(self.state_dict(), self.chpt_file+flag)

    def load_checkpoint(self, flag=""):
        print("... loading network checkpoint ..." )
        self.load_state_dict(T.load(self.chpt_file+flag))


class Agent(object):
    def __init__(self,
                 name,
                 net_dir,
                 \
                 gamma, lr,
                 \
                 input_dims,  n_actions,
                 depth, width, activ,
                 \
                 eps_clip, pol_sync,
                 \
                 env_name,
                 n_workers, n_frames,
                 vf_coef, ent_coef,
                 ):

        ## Setting all class variables
        self.__dict__.update(locals())
        self.learn_step_counter = 0

        ## The actor and critic networks are initialised
        self.actor_critic = ActorCriticMLP( self.name + "_ac_networks", net_dir,
                                            input_dims, n_actions,
                                            depth, width, activ )
        self.actor_critic_old = ActorCriticMLP( self.name + "_ac_networks", net_dir,
                                                input_dims, n_actions,
                                                depth, width, activ )
        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())

        ## The gradient descent algorithm and loss function used to train the policy network
        self.optimiser = optim.Adam( self.actor_critic.parameters(), lr = lr )

        ## We create the memory to hold the update for each batch
        self.memory = mm.SmallMemory( n_workers*n_frames, input_dims )
        self.memory.reset()

        ## We now initiate the list of workers
        self.workers = [ Worker(env_name, self, n_frames, gamma) for _ in range(n_workers) ]

        ## We also initiate the graph, which is an agent attribute as it is called by all workers
        plt.ion()
        self.sp = score_plot("PPO")

    def save_models(self, flag=""):
        self.actor_critic.save_checkpoint(flag)
        self.actor_critic_old.save_checkpoint(flag)

    def load_models(self, flag=""):
        self.actor_critic.load_checkpoint(flag)
        self.actor_critic_old.save_checkpoint(flag)

    def store_transition(self, state, action, value):
        ## Interface to memory, so no outside class directly calls it
        self.memory.store_transition(state, action, value)

    def choose_action(self, state):
        ## First we convert the state observation into a tensor
        state_tensor = T.tensor( [state], device=self.actor_critic.device, dtype=T.float32 )

        ## We then calculate the probabilities of taking each action
        policy = self.actor_critic_old.get_policy( state_tensor )

        ## To sample an action using these probabilities we use the distributions package
        action_dist   = T.distributions.Categorical(policy)
        chosen_action = action_dist.sample()

        return chosen_action.item()

    def train(self):

        ## First we have each worker fill up its portion of the batch
        self.memory.reset()
        for worker in self.workers:
            worker.fill_batch()

        ## We need to convert all of these arrays to pytorch tensors
        states  = T.tensor(self.memory.states).to(self.actor_critic.device)
        actions = T.tensor(self.memory.actions).to(self.actor_critic.device)
        values  = T.tensor(self.memory.values).to(self.actor_critic.device).reshape(-1,1)

        for _ in range(self.pol_sync):

            ## We zero out the gradients, as required for each pyrotch train loop
            self.optimiser.zero_grad()

            ## We start with the critic/value  loss
            ## We need both the value and policy based on the current states
            policy_old, state_values_old = self.actor_critic_old(states)

            ## We use the td_error as an estimator of the advantage value and the critic loss
            td_errors   = values - state_values_old
            critic_loss = td_errors.pow(2).mean()

            ## Now we move onto the actor/policy loss
            ## We need the to evaluate the actions taken using the new policy net
            policy, state_values = self.actor_critic(states)

            ## We need the importance sampling ratios of the actions taken
            action_dist = T.distributions.Categorical(policy)
            log_probs = action_dist.log_prob(actions).view(-1, 1)

            action_dist_old = T.distributions.Categorical(policy)
            log_probs_old = action_dist_old.log_prob(actions).view(-1, 1)

            ratios = T.exp( log_probs - log_probs_old.detach() )

            ## Now we can fund the loss
            surr1 = ratios * td_errors.detach()
            surr2 = T.clamp( ratios, 1-self.eps_clip, 1+self.eps_clip ) * td_errors.detach()
            actor_loss = - T.min( surr1, surr2 ).mean()

            ## We finally use the distribution to get the regularising entropy
            entropy = action_dist.entropy().mean()

            ## We do a single update step using the sum of losses (equivalent to two steps)
            loss = actor_loss + ( self.vf_coef * critic_loss ) - ( self.ent_coef * entropy )
            loss.backward()
            nn.utils.clip_grad_value_( self.actor_critic.parameters(), 1.0 )
            self.optimiser.step()

            self.learn_step_counter += 1

        ## We check if it is time to sync the policy networks
        self.actor_critic_old.load_state_dict( self.actor_critic.state_dict() )

        return loss.item()
