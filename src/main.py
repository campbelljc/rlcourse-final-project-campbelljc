from env import *
from agents import *
from policy import *
from gym_nethack import *

from itertools import product

if __name__ == '__main__':
    #env, num_episodes, max_num_steps_per_episode = ChainMDP(), 1000, 1000
    #env, num_episodes, max_num_steps_per_episode = Blackjack(), 5000, 100
    env, num_episodes, max_num_steps_per_episode = NetHackCombatGiantBatVsArrow(), 3000, 100
    
    num_evaluations = 50
    num_episodes_per_evaluation = 10
    
    #'''
    parameters = {
        'agent':        [LinearFAQAgent, LinearFASarsaAgent],
        'expl':         [None, HashCounting, DensityModel],
        'gamma':        [0.9, 0.99], # gamma
        'alpha':        [0.01, 0.1], # alpha
        'beta_density': [0.001, 0.01, 0.1], # beta_density
        'beta_hash':    [0.01, 0.1, 0.2], # beta_hash
        'k_hash':       [4, 16, 64, 128] # k_hash
    }
    '''
    parameters = {
        'agent':   [TabularQAgent, DelayedQAgent, DelayedQIEAgent], # TabularSarsaAgent
        'gamma':   [0.9, 0.99, 0.995],
        'alpha':   [0.01, 0.1],
        'm':       [1, 10, 100, 1000],
        'e1':      [0.001, 0.1],
        'b':       [1, 2, 3]
    }
    '''
    
    param_combos = list(product(*parameters.values()))
    param_names = list(parameters)
    
    learning_policy = LinearAnnealedPolicy
    learning_inner_policy = EpsGreedyPossibleQPolicy #EpsGreedyQPolicy
    value_max = 0.25
    value_min = 0.05
    nb_steps = num_episodes
    
    # ref for eps rate: https://gym.openai.com/evaluations/eval_32cSSqJMTIi12bfyPk6eQ
    
    eval_policy = GreedyPossibleQPolicy()
    
    for j, combo in enumerate(param_combos):
        params = dict(zip(param_names, combo))
        l_policy = learning_policy(inner_policy=learning_inner_policy(), attr='eps', value_max=value_max, value_min=value_min, value_test=value_min, nb_steps=nb_steps)
        print(params)
        env.reset()
        agent_fn = params['agent']
        agent = agent_fn(env=env, policy=l_policy, **params)
        eval_agent = agent_fn(env=env, policy=eval_policy, **params)
        eval_agent.set_testing()
        
        if os.path.exists(agent.savedir + '/records.dll'): continue
        
        for ep in range(num_episodes+1):
            print("Combo", j, "episode", ep)
            #agent.reset()
            agent.new_episode()
            state = env.new_episode()
            agent.prev_state = state
            while agent.cur_action_this_episode < max_num_steps_per_episode:
                action = agent.choose_action(state)
                state, reward, terminal = env.step(action)
                agent.update(state, reward, terminal)
                agent.end_turn(state)
                if terminal:
                    break
            agent.end_episode()
            env.end_episode()
            
            #agent.plot_frame()
            if ep % (num_episodes / num_evaluations) == 0:
                # time to evaluate target policy.
                for eval_ep in range(num_episodes_per_evaluation):
                    # copy the q-values. (start each ep fresh)
                    eval_agent.set_params(*agent.get_params())
        
                    print("Agent", j, ", episode", eval_ep, "(evaluation)")
                    eval_agent.new_episode()
                    state = env.new_episode()
                    eval_agent.prev_state = state
                    while eval_agent.cur_action_this_episode < max_num_steps_per_episode:
                        action = eval_agent.choose_action(state)
                        state, reward, terminal = env.step(action)
                        eval_agent.update(state, reward, terminal, update=False)
                        eval_agent.end_turn(state)
                        if terminal:
                            break
                    eval_agent.end_episode()
                    env.end_episode()
                eval_agent.avg_last_episodes(ep, num_episodes_per_evaluation)
        eval_agent.save_stats()
        env.save_stats()
