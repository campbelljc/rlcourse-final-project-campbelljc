from env import *
from fileio import *
from gym_nethack import *
from agents import Episode

import os
import numpy as np
import dill, pickle
import seaborn as sns
import matplotlib.pyplot as plt 

envs = [
    #Blackjack()
    #CartPole(),
    #ChainMDP(),
    #StrehlBanditMDP(),
    #StrehlHallwaysMDP(),
    NetHackCombatGiantBatVsArrow()
]

if not os.path.exists('../figs'):
    os.makedirs('../figs')

graph_split_fns = [('tabular', lambda name: 'Tabular' in name or 'Delayed' in name), ('linear', lambda name: 'Linear' in name)]

plt.rcParams['savefig.dpi'] = 500

for env in envs:
    for graph_split_name, graph_split_fn in graph_split_fns:    
        # load records
        model_records = []
        for subdir in get_immediate_subdirectories('../results'):
            if env.name not in subdir: continue
            if not os.path.exists("../results/"+subdir+"/records.dll"): continue

            # load the results for this model
            records = dill.load(open("../results/"+subdir+"/records.dll", 'rb'))
            model_name = subdir[len(env.name)+7:]
            if not graph_split_fn(model_name): continue
            model_records.append((model_name, records))
    
        # plot the avg total reward per model
        avg_total_rewards = []
        for model_name, records in model_records:
            avg_total_reward = sum([rec.total_reward for rec in records])/len(records)
            avg_total_rewards.append((model_name, avg_total_reward))
    
        sns.despine()
        plt.figure()
        fig, axes = plt.subplots()
        fig.set_size_inches(10, 45)
        plt.barh(np.arange(len(avg_total_rewards)), [r[1] for r in avg_total_rewards])
        plt.yticks(np.arange(len(avg_total_rewards)), [r[0] for r in avg_total_rewards])
        plt.xlabel('Average total reward')
        plt.title(env.name)
        plt.tight_layout()
        plt.savefig('../figs/' + env.name + '_' + graph_split_name + '_avgtotalreward.png')
        plt.close()
    
        # plot the reward per episode over time for each model
        num_recs = 0
        best_results = defaultdict(lambda: (-np.inf, None, None))
    
        plt.figure()
        for model_name, records in model_records:
            avg_total_reward = sum([rec.total_reward for rec in records])/len(records)
            plt.plot([rec.game_id for rec in records], [rec.total_reward for rec in records], label=model_name)
        
            model_name_stripped = model_name.split("_")[0]
            if best_results[model_name_stripped][0] < avg_total_reward:
                best_results[model_name_stripped] = (avg_total_reward, records, model_name)
                num_recs += 1
    
        plt.legend()
        plt.ylabel('Total reward over time')
        plt.title(env.name)
        plt.savefig('../figs/' + env.name + '_' + graph_split_name + '_totalreward.png')
        plt.close()
    
        # plot the reward per episode over time for each model (best parameter models only)
        sns.set_palette(sns.color_palette("hls", len(best_results)))
        plt.figure()                
        for model_name_stripped in best_results:
            avg_total_reward, records, model_name = best_results[model_name_stripped]
            plt.plot([rec.game_id for rec in records], [rec.total_reward for rec in records], label=model_name)
    
        plt.legend()
        plt.ylabel('Total reward over time')
        plt.title(env.name)
        plt.savefig('../figs/' + env.name + '_' + graph_split_name + '_totalreward_bestparams.png')
        plt.close()
    
        # plot the avg total reward per model (best parameter models only)
        #colors = ['yellow', 'orange', 'red', 'brown', 'maroon', 'silver']
        sns.despine()
        plt.figure()
        fig, axes = plt.subplots()
        if graph_split_name == 'linear':
            fig.set_size_inches(8, 3)
        else:
            fig.set_size_inches(8, 2)
        
        plt.barh(np.arange(len(best_results)), [best_results[model][0] for model in best_results]) #, color=colors)
        plt.yticks(np.arange(len(best_results)), [best_results[model][2] for model in best_results])
        plt.xlabel('Average total reward')
        plt.title(env.name)
        plt.tight_layout()
        plt.savefig('../figs/' + env.name + '_' + graph_split_name + '_avgtotalreward_bestparams_final.png')
        plt.close()
    
        # plot total cumulative reward per model
        sns.set_palette(sns.color_palette("hls", len(model_records)))
        plt.figure()
        for model_name, records in model_records:
            cumulative_reward = np.cumsum([rec.total_reward for rec in records])
            plt.plot([rec.game_id for rec in records], cumulative_reward, label=model_name)
        
        plt.legend()
        plt.ylabel('Total cumulative reward over time')
        plt.title(env.name)
        plt.savefig('../figs/' + env.name + '_' + graph_split_name + '_totalcreward.png')
        plt.close()
    
        ## plot state counts for each model
        #if not os.path.exists('../figs/state_counts'):
        #    os.makedirs('../figs/state_counts')
    
        # (for nh env only) - get possible state configurations.
        #           mon dist   4xdiststuff   hp  clvl/ac equip stateff nummons inven  ammo lof
        num_states = 1 * 18 * 2 * 2 * 2 * 2 * 5 * 1 * 1 * 2    * 1     * 1     *  1  * 2 * 2
    
        # load the state counts into the list below, and also create a set of all visited states.
        state_counts_per_model = []
        states = set()
        best_results_vars = {}
        variances = []
        for subdir in get_immediate_subdirectories('../results'):
            if env.name not in subdir: continue
            if not os.path.exists("../results/"+subdir+"/records.dll"): continue
        
            model_name = subdir[len(env.name)+7:]
            if not graph_split_fn(model_name): continue
            with open("../results/"+subdir+"/state_counts.pkl", 'rb') as input:
                state_counts = pickle.load(input)
                variance = np.var(list(state_counts.values()))
                state_counts_per_model.append((model_name, state_counts, variance))
            
                variances.append((variance, model_name))
            
                model_name_stripped = model_name.split("_")[0]                
                if model_name_stripped in best_results and len(state_counts.values()) > 0:
                    # store the variance of this one.
                    best_results_vars[model_name_stripped] = (*best_results[model_name_stripped], variance)
        
            for state in state_counts:
                states.add((state))
    
        # plot all variances
        sns.despine()
        plt.figure()
        fig, axes = plt.subplots()
        fig.set_size_inches(10, 45)
        plt.barh(np.arange(len(variances)), [r[0] for r in variances])
        plt.yticks(np.arange(len(variances)), [r[1] for r in variances])
        plt.xlabel('State count variance per model')
        plt.title(env.name)
        plt.tight_layout()
        plt.savefig('../figs/' + env.name + '_' + graph_split_name + '_scountvariance.png')
        plt.close()
    
        # plot the avg total reward per model (best parameter models only)
        sns.despine()
        plt.figure()
        plt.barh(np.arange(len(best_results_vars)), [best_results_vars[model][3] for model in best_results_vars])
        plt.yticks(np.arange(len(best_results_vars)), [best_results_vars[model][2] for model in best_results_vars])
        plt.xlabel('State count variance of best models')
        plt.title(env.name)
        plt.tight_layout()
        plt.savefig('../figs/' + env.name + '_' + graph_split_name + '_scountvariance_bestparams_final.png')
        plt.close()
        
        '''
        states = list(states)
        for model_name, state_counts, variance in state_counts_per_model:
            # create the counts array...
            counts = []
            for state in states:
                if state in state_counts:
                    counts.append(state_counts[state])
                else:
                    counts.append(0)
        
            plt.figure()
            plt.ylim(0, 4000)
            plt.plot([i for i in range(len(counts))], counts, label=model_name)        
            plt.legend()
            plt.ylabel('Counts per state')
            plt.title(env.name)
            plt.savefig('../figs/state_counts/' + '_' + graph_split_name + env.name + '_' + model_name + '_scounts.png')
            plt.close()
        
            # ref for later:
            # http://stackoverflow.com/questions/15740682/wrapping-long-y-labels-in-matplotlib-tight-layout-using-setp
        '''