import gym
import sys
import json

from itertools import count

from Algorithms import D3QN, IQN

def main():

    ########## User settings ##########

    env_name = 'LunarLander-v2'
    alg  = 'IQN'

    name = 'Test'
    save_dir = 'Saved_Models'

    test_mode = False ## No training or exploration
    load_prev = False ## Load a checkpoint with the same name
    save_every = 1000 ## Number of environment steps before save checkpoint is called
    render_env = True ## Render the environment and the rewards in their own windows
    render_rtn = 2    ## Number of environment steps before rendering expected return

    ## Additional kwargs for the algorithm
    alg_kwargs = {
        'n_quantiles':8,
    }

    ###################################

    ## Initialise the environment
    env = gym.make(env_name)

    ## Collect all of the kwargs and configs into a single dictionary
    all_kwargs = { 'name': name,
                   'save_dir': save_dir,
                   'env': env,
                   'alg_kwargs': alg_kwargs}
    with open('OffPolicy_Config.json') as file:
        all_kwargs.update(json.load(file))

    ## Initialise the agent
    if   alg == 'D3QN':  agent = D3QN.D3QN_Agent(**all_kwargs)
    elif alg == 'IQN':   agent = IQN.IQN_Agent(**all_kwargs)

    ## Load model checkpoint
    if load_prev:
        agent.load_checkpoint()

    ## Put the agent into test mode, which makes it always do greedy actions
    agent.test_mode = test_mode

    ## Create plots for visualising episode rewards and state value predictions
    # plt.ion()
    # sp = myPT.score_plot(agent.name)
    # if draw_return:
    #     if alg == 'D3QN': vp = myPT.value_plot()
    #     if alg == 'IQN': vp = myPT.quant_plot()

    ## Initial values
    all_time = 0
    loss = 0

    ## The episode loop
    for ep in count():

        ## Resetting the environment and episode stats
        state = env.reset()
        ep_score = 0.0
        ep_loss  = 0.0

        ## Running through an episode
        for t in count():

            ## Visualise the environment
            if render_env:
                env.render()

            ## The agent chooses an action
            action, value = agent.choose_action(state)

            ## The environment evolves wrt chosen action
            next_state, reward, done, info = env.step(action)

            ## The transition is stored into memory and the model is trained
            if not test_mode:
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train()

            ## The state is replaced
            state = next_state

            ## Printing running episode scores
            all_time += 1
            ep_score += reward
            ep_loss  += loss
            eps = agent.eps
            print('Score = {:.7} \r'.format(ep_score), end='')
            sys.stdout.flush()

            # ## Plot the expected return
            # if draw_return and all_time%draw_interv==0:
            #     vp.update(value)
            #
            # ## Save a checkpoint for the models
            # if not test_mode and all_time%save_every==0:
            #     agent.save_models()

            ## Break the episode loop if the environment returns a terminal state
            if done:
                break

        ## Prining and plotting the completed episode stats
        # sp.update(ep_score)
        ep_loss /= (t+1)
        print( 'Episode {}: Reward = {:.7}, Loss = {:4.3f}, Eps = {:4.3f}, Episode Time = {}, Total Time = {}'.format( \
                                         ep, ep_score, ep_loss, eps, t+1, all_time))


if __name__ == '__main__':
    main()
