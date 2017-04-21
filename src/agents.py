from policy import GreedyPossibleQPolicy
from fileio import append, write_list, read_list
from density import LocationDependentDensityModel

import os
import numpy as np
import dill, pickle
from math import sqrt
from copy import deepcopy
from functools import partial
from collections import namedtuple, defaultdict

Episode = namedtuple('Episode', 'game_id num_actions total_reward')

class Agent():
    def __init__(self):
        self.training = True
        self.cur_episode = 0
        self.cur_action = 0
        self.game_records = []        
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
    
    def set_testing(self):
        self.training = False
    
    def save_stats(self):
        write_list([self.cur_episode, self.cur_action], self.savedir+"stats")
        with open(self.savedir+"records.dll", 'wb') as output:
            dill.dump(self.game_records, output)
    
    def new_episode(self):
        self.total_reward = 0
        self.cur_action_this_episode = 0
    
    def end_turn(self):
        self.cur_action += 1
        self.cur_action_this_episode += 1
    
    def end_episode(self):
        if not self.training:
            self.game_records.append(Episode(self.cur_episode, self.cur_action_this_episode, self.total_reward))
        print(self.cur_episode, self.cur_action_this_episode, self.total_reward)
        self.cur_episode += 1
    
    def avg_last_episodes(self, eval_id, num_to_avg):
        records = self.game_records[-num_to_avg:]
        self.game_records = self.game_records[:-num_to_avg]
        
        avg_actions = sum([rec.num_actions for rec in records])/len(records)
        avg_total_reward = sum([rec.total_reward for rec in records])/len(records)
        self.game_records.append(Episode(eval_id, avg_actions, avg_total_reward))

class QAgent(Agent):
    def __init__(self, env, policy, gamma, name, tabular):
        self.policy = policy
        if policy is not None:
            self.policy._set_agent(self)
        self.gamma = gamma
        self.num_actions = env.num_actions
        self.state_length = env.state_length
        self.tabular = tabular
        self.env = env
        self.env._set_agent(self)
        
        self.savedir = '../results/model_' + env.name + '_' + name + '_' + self.get_param_str() + '/'
        super().__init__()
        self.reset()
    
    def reset(self):
        self.state_counts = defaultdict(int)
    
    def get_param_str(self):
        return 'g' + str(self.gamma)
    
    def save_stats(self):
        super().save_stats()    
        write_list([self.alpha, self.gamma], self.savedir+"params")
        with open(self.savedir+"state_counts.pkl", 'wb') as output:
            pickle.dump(self.state_counts, output)
    
    def new_episode(self):
        super().new_episode()
        self.state = None
        self.prev_state = None
        self.last_action = None
        
    def end_turn(self, state):
        self.prev_state = np.copy(state)
        super().end_turn()
        
    def choose_action(self, state):
        action = self.policy.select_action(q_values=self.get_qvals_for_state(state))
        self.last_action = action
        return action
        
    def observe(self, state, reward, terminal, update=True):
        assert self.last_action is not None
        assert self.prev_state is not None
        
        self.total_reward += reward
        
        if not update:
            return
        
        self.state_counts[(tuple(self.prev_state), self.last_action)] += 1

class TabularQAgent(QAgent):
    def __init__(self, env, policy, alpha, gamma, name='QLearningTabular', **args):
        self.alpha = alpha
        super().__init__(env, policy, gamma, name, tabular=True)
    
    def reset(self):
        super().reset()
        self.qvals = defaultdict(lambda: [1/(1-self.gamma)] * self.env.num_actions)
        if 'get_initial_qvalues' in dir(self.env):
            self.qvals = self.env.get_initial_qvalues()
        
    def get_param_str(self):
        return super().get_param_str() + '_a' + str(self.alpha)
    
    def get_params(self):
        return self.qvals, self.state_counts #[deepcopy(self.qvals), deepcopy(self.state_counts)]
    
    def set_params(self, qvals, state_counts):
        self.qvals = qvals
        self.state_counts = state_counts
    
    def get_qvals_for_state(self, state):
        return self.qvals[tuple(state)]
    
    def predict(self, state):
        return max(self.qvals[tuple(state)])
    
    def update(self, next_state, reward, terminal, update=True):
        self.observe(next_state, reward, terminal, update)
        
        if not update:
            return
        
        #print(next_state)
        update = reward # tt = rr
        if not terminal:
            update += self.gamma * self.predict(next_state)
        
        alpha = 1/((self.state_counts[(tuple(self.prev_state), self.last_action)] + 1) ** self.alpha)
        
        self.qvals[tuple(self.prev_state)][self.last_action] += alpha * (update - self.qvals[tuple(self.prev_state)][self.last_action])

class DelayedQAgent(TabularQAgent):
    def __init__(self, env, policy, gamma, m, e1, name='DelayedQ', **args):
        self.m = m
        self.e1 = e1
        assert 0 <= self.e1 <= 1
        
        super().__init__(env, policy, alpha=0, gamma=gamma, name=name)
        
    def reset(self):
        super().reset()
        self.delayed_updates = defaultdict(int) # : 0 -- (state, action) tuple as indices (U(s,a))
        self.state_act_counter = defaultdict(int) # defaults: 0 # (I(s,a))
        self.update_timestep_starts = defaultdict(int) # : 0    # (b(s,a))
        self.learn_flags = defaultdict(lambda: True) # (s, a) tuples -> True initially # LEARN(s,a)
        self.timestep_of_latest_qval_change = 0
    
    def get_param_str(self):
        return super().get_param_str() + '_m' + str(self.m) + '_e' + str(self.e1)
    
    def update(self, next_state, reward, terminal, update=True):
        self.observe(next_state, reward, terminal, update)
                
        if not update:
            return
                        
        #if self.update_timestep_starts[(self.prev_state, action)] <= self.timestep_of_latest_qval_change:
        #    self.learn_flags[(self.prev_state, action)] = True
        
        if self.learn_flags[(tuple(self.prev_state), self.last_action)]:
            #if self.state_act_counter[(self.prev_state, action)] == 0:
            #    self.update_timestep_starts[(self.prev_state, action)] = self.cur_action_num # t val
            
            update = reward
            if not terminal:
                update += (self.gamma * self.predict(next_state))
            
            # accumulate the updates (will average later, when update is applied)
            self.delayed_updates[(tuple(self.prev_state), self.last_action)] += update
            self.state_act_counter[(tuple(self.prev_state), self.last_action)] += 1
            
            if self.state_act_counter[(tuple(self.prev_state), self.last_action)] == self.m:
                averaged_q_target = (self.delayed_updates[(tuple(self.prev_state), self.last_action)] / self.m)
                if self.qvals[tuple(self.prev_state)][self.last_action] - averaged_q_target >= 2 * self.e1:
                    self.qvals[tuple(self.prev_state)][self.last_action] = averaged_q_target + self.e1
                    self.timestep_of_latest_qval_change = self.cur_action
                elif self.update_timestep_starts[(tuple(self.prev_state), self.last_action)] >= self.timestep_of_latest_qval_change:
                    self.learn_flags[(tuple(self.prev_state), self.last_action)] = False
                self.update_timestep_starts[(tuple(self.prev_state), self.last_action)] = self.cur_action
                self.delayed_updates[(tuple(self.prev_state), self.last_action)] = 0
                self.state_act_counter[(tuple(self.prev_state), self.last_action)] = 0
        elif self.update_timestep_starts[(tuple(self.prev_state), self.last_action)] < self.timestep_of_latest_qval_change:
            self.learn_flags[(tuple(self.prev_state), self.last_action)] = True

class DelayedQIEAgent(DelayedQAgent):
    def __init__(self, env, policy, gamma, m, e1, b, name='DelayedQIE', **args):
        super().__init__(env, policy, gamma, m, e1, name)
        self.b = b
    
    def update(self, next_state, reward, terminal, update=True):
        self.observe(next_state, reward, terminal, update)
        
        if not update:
            return
                
        if self.update_timestep_starts[(tuple(self.prev_state), self.last_action)] <= self.timestep_of_latest_qval_change:
            self.learn_flags[(tuple(self.prev_state), self.last_action)] = True
        
        if self.learn_flags[(tuple(self.prev_state), self.last_action)]:
            if self.state_act_counter[(tuple(self.prev_state), self.last_action)] == 0:
                self.update_timestep_starts[(tuple(self.prev_state), self.last_action)] = self.cur_action # t val
                self.delayed_updates[(tuple(self.prev_state), self.last_action)] = 0
            self.state_act_counter[(tuple(self.prev_state), self.last_action)] += 1
            
            if terminal:
                update = reward
            else:
                newQ = self.qvals[tuple(next_state)]
                maxQ = np.max(newQ)
                update = reward + (self.gamma * maxQ)
            
            # accumulate the updates (will average later, when update is applied)
            self.delayed_updates[(tuple(self.prev_state), self.last_action)] += update
            
            averaged_q_target = (self.delayed_updates[(tuple(self.prev_state), self.last_action)] / self.state_act_counter[(tuple(self.prev_state), self.last_action)]) + (self.b / sqrt(self.state_act_counter[(tuple(self.prev_state), self.last_action)]))
            if self.qvals[tuple(self.prev_state)][self.last_action] - averaged_q_target >= self.e1:
                self.qvals[tuple(self.prev_state)][self.last_action] = averaged_q_target
                self.timestep_of_latest_qval_change = self.cur_action
                self.state_act_counter[(tuple(self.prev_state), self.last_action)] = 0
            elif self.state_act_counter[(tuple(self.prev_state), self.last_action)] == self.m:
                self.state_act_counter[(tuple(self.prev_state), self.last_action)] = 0
                if self.update_timestep_starts[(tuple(self.prev_state), self.last_action)] > self.timestep_of_latest_qval_change:
                    self.learn_flags[(tuple(self.prev_state), self.last_action)] = False

class TabularSarsaAgent(TabularQAgent):
    def __init__(self, env, policy, alpha, gamma, name='SarsaTabular', **args):
        super().__init__(env, policy, alpha, gamma, name)
        self.greedy_policy = GreedyPossibleQPolicy()
        self.greedy_policy._set_agent(self)
    
    def new_episode(self):
        super().new_episode()
        self.next_action = None
    
    def best_action(self, state):
        return self.greedy_policy.select_action(q_values=self.get_qvals_for_state(state))
    
    def choose_action(self, state):
        assert self.next_action is not None or self.cur_action_this_episode is 0
        if self.cur_action_this_episode is 0:
            action = self.policy.select_action(q_values=self.get_qvals_for_state(state))
        else:
            action = self.next_action
            self.next_action = None
        self.last_action = action
        return action
    
    def update(self, next_state, reward, terminal, update=True):
        self.observe(next_state, reward, terminal, update)
        assert self.next_action is None
        
        if not terminal:
            self.next_action = self.best_action(next_state)
        
        if not update:
            return
        
        update = reward # tt = rr
        if not terminal:
            update += self.gamma * self.qvals[tuple(next_state)][self.next_action]
        
        alpha = 1/((self.state_counts[(tuple(self.prev_state), self.last_action)] + 1) ** self.alpha)
                
        self.qvals[tuple(self.prev_state)][self.last_action] += alpha * (update - self.qvals[tuple(self.prev_state)][self.last_action])

class LinearFAQAgent(QAgent):
    def __init__(self, env, policy, expl, alpha, gamma, name='LinearQ', **args):
        self.alpha = alpha
        if expl is None:
            self.expl = None
        else:
            self.expl = expl(env, **args)
            name += self.expl.name
        super().__init__(env, policy, gamma, name, tabular=False)
        self.num_features = env.num_features
        self.weights = np.random.normal(0, 0.1, self.num_features)
    
    def get_param_str(self):
        pstr = super().get_param_str() + '_a' + str(self.alpha)
        if self.expl is not None:
            pstr += self.expl.get_param_str()
        return pstr

    def get_params(self):
        return [np.copy(self.weights)]
    
    def set_params(self, weights):
        self.weights = weights
    
    def get_qvals_for_state(self, state):
        return [self.get_qvalue(state, i) for i in range(self.num_actions)]
    
    def best_action(self, state):
        return np.argmax([self.get_qvalue(state, act) for act in range(self.num_actions)])
    
    def get_qvalue(self, state, action):
        return np.dot(self.weights, self.env.get_features_for_state_action(state, action))
        #return np.sum(self.weights[self.env.get_features_for_state_action(state, action)])
    
    def get_reward_bonus(self, prev_state, action):
        if self.expl is None:
            return 0
        return self.expl.get_reward_bonus(prev_state, action)
    
    def update(self, next_state, reward, terminal, update=True):
        assert self.last_action is not None
        assert self.prev_state is not None
        
        self.total_reward += reward
        
        if not update:
            return
        
        action = self.last_action
        
        predicted_qvalue = self.get_qvalue(self.prev_state, action)
        
        reward += self.get_reward_bonus(self.prev_state, action)        
        if terminal:
            td_error = reward - predicted_qvalue
        else:
            best_next_action = self.best_action(next_state)
            predicted_next_qval = self.get_qvalue(next_state, best_next_action)
            td_error = reward + (self.gamma * predicted_next_qval) - predicted_qvalue
                
        state_features = self.env.get_features_for_state_action(self.prev_state, action)
        #ew = np.zeros((self.num_features,))
        #for feat in state_features:
        #    ew[feat] = 1
        
        self.weights += np.multiply(self.alpha * td_error, state_features)

class LinearFASarsaAgent(LinearFAQAgent):
    def __init__(self, env, policy, expl, alpha, gamma, name='LinearSarsa', **args):
        super().__init__(env, policy, expl, alpha, gamma, name, **args)
        self.greedy_policy = GreedyPossibleQPolicy()
        self.greedy_policy._set_agent(self)
    
    def new_episode(self):
        super().new_episode()
        self.next_action = None
    
    def best_action(self, state):
        return self.greedy_policy.select_action(q_values=self.get_qvals_for_state(state))
    
    def choose_action(self, state):
        assert self.next_action is not None or self.cur_action_this_episode is 0
        if self.cur_action_this_episode is 0:
            action = self.policy.select_action(q_values=self.get_qvals_for_state(state))
        else:
            action = self.next_action
            self.next_action = None
        self.last_action = action
        return action
    
    def update(self, next_state, reward, terminal, update=True):
        assert self.last_action is not None
        assert self.prev_state is not None
        assert self.next_action is None
        
        self.total_reward += reward
        
        if not terminal:
            self.next_action = self.best_action(next_state)
        
        if not update:
            return
        
        update = reward + self.get_reward_bonus(self.prev_state, self.last_action)
        if not terminal:
            update += self.gamma * self.get_qvalue(next_state, self.next_action)
        
        state_features = self.env.get_features_for_state_action(self.prev_state, self.last_action)
        #ew = np.zeros((self.num_features,))
        #for feat in state_features:
        #    ew[feat] = 1
        
        self.weights += np.multiply(self.alpha * (update - self.get_qvalue(self.prev_state, self.last_action)), state_features)

class DensityModel():
    def __init__(self, env, beta_density, **args):
        self.name = 'Density'
        self.env = env
        self.beta = beta_density
        self.density_model = LocationDependentDensityModel(state_length=self.env.num_state_features, state_splits=self.env.get_density_state_splits())
        # context_functor=self.full_ctxt)
    
    def full_ctxt(self, frame, y):
        context = [0] * (len(frame) - 1)
    
        index = 0
        for i in range(len(frame)):
            if index == i: continue
            context[index] = frame[i]
            index += 1
        
        return context
    
    def get_param_str(self):
        return '_b' + str(self.beta)
    
    def get_reward_bonus(self, prev_state, action):
        feats = self.env.get_features_for_state(prev_state)
        
        #feats = [0] * self.env.num_state_features
        #for i in state_features:
        #    feats[i] = 1
        
        # update density model & get recoding prob
        cur_prob = self.density_model.update(feats) #np.array(self.prev_state))
        recoding_prob = self.density_model.log_prob(feats) #np.array(self.prev_state))
        
        if cur_prob == recoding_prob or recoding_prob < cur_prob:
            return 0
        
        pseudo_count = (cur_prob * (1 - recoding_prob)) / (recoding_prob - cur_prob)
        if pseudo_count < 0:
            # ref: https://github.com/brendanator/atari-rl/blob/7764b85f25100195dbba32ad4490261b277efdbe/agents/exploration_bonus.py
            pseudo_count = 0 # "occasionally happens at start of training"
        
        exploration_bonus = self.beta / sqrt(pseudo_count + 0.01)
        #print("Exp bonus:", exploration_bonus, "from pseudo-count", pseudo_count)
        return exploration_bonus

class HashCounting():
    def __init__(self, env, beta_hash, k_hash, **args):
        self.name = 'HashCounting'
        self.env = env
        self.beta = beta_hash
        self.k = k_hash
        self.A = np.zeros((self.k, self.env.num_features)) # [[np.random.normal(0, 1) for i in range(16)] for j in range(k)]
        self.hashed_state_counts = defaultdict(lambda: 0)
    
    def get_param_str(self):
        return '_b' + str(self.beta) + '_k' + str(self.k)
    
    def get_reward_bonus(self, prev_state, action):
        feats = self.env.get_features_for_state_action(prev_state, action)
        #feats = np.zeros((self.env.num_features,))
        #for feat in state_features:
        #    feats[feat] = 1
        
        # compute hash of current state
        state_hash = np.sign(np.dot(self.A, feats))
        self.hashed_state_counts[tuple(state_hash)] += 1
        exploration_bonus = self.beta / sqrt(self.hashed_state_counts[tuple(state_hash)])
        return exploration_bonus
