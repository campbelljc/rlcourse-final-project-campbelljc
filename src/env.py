from easy21feature import dealerFeatures, playerFeatures

import gc
import gym
import dill
import random
import numpy as np
from collections import defaultdict

class Env():
    def __init__(self, name, num_actions, state_length, num_features=None, discrete=False):
        self.name = name
        self.discrete = discrete
        self.num_actions = num_actions
        self.state_length = state_length
        self.num_features = state_length if num_features is None else num_features
    
    def _set_agent(self, agent):
        self.agent = agent
    
    def new_episode(self):
        pass
    
    def end_episode(self):
        pass
    
    def should_end_episode(self):
        return False
    
    def reset(self):
        pass
    
    def save_stats(self):
        pass
    
    def step(self, action):
        raise NotImplementedError
    
class StrehlBanditMDP(Env):
    def __init__(self):
        super().__init__('strehl_banditMDP', num_actions=6, state_length=1)
    
    def new_episode(self):
        self.state = [0]
        return self.state
    
    def get_initial_qvalues(self):
        return defaultdict(lambda: [1/(1-0.95)] * self.num_actions)
    
    def step(self, action):
        action += 1 # 6 actions {0 .. 5 } -> {1 .. 6}
        if self.state[0] == 0:
            state_transitions = [action, 0]
            transition_probs  = [1/action, 1 - (1/action)]
            rewards           = [0] * self.num_actions
        elif self.state[0] > 0:
            state_transitions = [0] # all actions lead to state 0
            transition_probs  = [1]
            rewards           = [(3/2)**self.state[0]] + ([0] * (self.num_actions-1))
        
        assert len(state_transitions) == len(transition_probs)
        assert len(rewards) == self.num_actions
        self.state = [np.random.choice(state_transitions, p=transition_probs)]
        return self.state, rewards[action-1], False

class StrehlHallwaysMDP(Env):
    def __init__(self):
        super().__init__('strehl_hallwaysMDP', num_actions=2, state_length=3, discrete=True)
        
    def new_episode(self):
        self.state = (-1, -1, -1)
        return self.state
    
    def step(self, action):
        if self.state[1] == -1: # no hallway specified
            if self.state[0] == -1: # start node
                if action == 0:
                    self.state = (0, -1, -1) # left node
                elif action == 1:
                    self.state = (1, -1, -1) # right node
            elif self.state[0] == 0: # left node
                if action == 0:
                    self.state = (0, 1, 0) # left-left node -> hallway 1
                elif action == 1:
                    self.state = (0, 2, 0) # left-right node -> hallway 2
            elif self.state[0] == 1: # right node
                if action == 0:
                    self.state = (0, 3, 0) # right-left node -> hallway 3
                elif action == 1:
                    self.state = (0, 4, 0) # right-right node -> hallway 4
            return self.state, 0, False
        else:
            if action == 0:
                cur_hallway = self.state[1]
                assert 1 <= cur_hallway <= 4
                if self.state[2] < 10:
                    # move forward one unit in hallway
                    self.state = (0, cur_hallway, self.state[2]+1)
                    return self.state, 0, False
                else:
                    # reached end, go back to beginning
                    self.state = (-1, -1, -1) # back to initial state
                    reward       = [0, (3/2)**(cur_hallway+5)]
                    reward_probs = [(cur_hallway-1)/cur_hallway, 1/cur_hallway]
                    return self.state, np.random.choice(reward, p=reward_probs), False
            elif action == 1:
                # no state change
                return self.state, 0, False

class ChainMDP(Env):
    def __init__(self):
        super().__init__('bayesian_chainMDP', state_length=10, num_actions=2, discrete=True)
    
    def new_episode(self):
        self.state = [0] * self.state_length
        self.state[0] = 1
        return self.state
    
    def step(self, action):
        assert action == 0 or action == 1
        assert sum(self.state) == 1 # can only be in one place at a time...
        
        if random.random() <= 0.2:
            action = 1 - action # "slip" and perform opposite action.
        
        if action == 1:
            # return to initial state and get reward of 2
            self.state = [0] * self.state_length
            self.state[0] = 1
            reward = 2
        elif action == 0:
            if self.state[-1] == 1:
                # in final state, so we get 10 reward & state stays the same
                reward = 10
            else:
                # proceed to next node (rightmost) in chain, get no reward
                reward = 0
                for i in self.state:
                    if self.state[i] == 1: # find where we are in the chain...
                        self.state[i]   = 0
                        self.state[i+1] = 1 # advance one
                        break
        
        return self.state, reward, False

class CartPole(Env):
    def __init__(self):
        self.gym = gym.make('CartPole-v0')
        
        # src: https://gym.openai.com/evaluations/eval_JeP6rWUQ8KuT8HB0YcR3g
        self.DIMS = 4
        self.N_TILES = 2 #5
        self.N_TILINGS = 2 #16
        self.TILES_START = np.array([-2.5, -4, -0.25, -3.75], dtype=np.float64)
        self.TILES_END   = np.array([ 2.5,  4,  0.25,  3.75], dtype=np.float64)
        self.TILES_RANGE = self.TILES_END - self.TILES_START
        self.TILES_STEP = (self.TILES_RANGE / (self.N_TILES * self.N_TILINGS))
        
        super().__init__('gym_cartpole', num_actions=2, num_features=self.N_TILINGS * (self.N_TILES ** self.DIMS) * 2, state_length=self.gym.observation_space.shape[0])
    
    def get_features_for_state_action(self, state, action=None):
        # src: https://gym.openai.com/evaluations/eval_JeP6rWUQ8KuT8HB0YcR3g
        indices = [np.floor(self.N_TILES*(state - self.TILES_START + (i * self.TILES_STEP))/self.TILES_RANGE).astype(np.int) for i in range(self.N_TILINGS)]

        flattened_indices = np.array([np.ravel_multi_index(index, dims=tuple([self.N_TILES] * self.DIMS), mode='clip') for index in indices])
        if action is not None:
            flattened_indices += int(action * (self.N_TILINGS * (self.N_TILES ** self.DIMS)))
        
        return flattened_indices
        
        feat_array = np.zeros((self.num_features,))
        for i in flattened_indices:
            feat_array[i] = 1
        
        #assert sum(feat_array) == len(flattened_indices)
        
        return feat_array #flattened_indices
    
    def new_episode(self):
        return self.gym.reset()
    
    def step(self, action):
        state, reward, done, _, _ = self.gym.step(action)
        return state, reward, done

class Blackjack(Env):
    def __init__(self):
        self.gym = gym.make('Blackjack-v0')
        self.num_state_features = 18
        
        super().__init__('gym_blackjack', num_actions=self.gym.action_space.n, num_features=36, state_length=-1)
    
    def get_features_for_state(self, state):
        tmp = np.zeros(shape=(3,6)) #zeros array of dim 3*6*2
        for i in dealerFeatures(state[1]): # range [0..2] (3 total)
            for j in playerFeatures(state[0]): # range [0..5] (6 total)
                tmp[i,j] = 1 # 2 actions total
        x = tmp.flatten() #(tmp.flatten()) #returning 'vectorized' (1-dim) array
        
        feats = [i for i, j in enumerate(x) if j == 1]
        return feats

    def get_features_for_state_action(self, state, action):
        # src: https://github.com/rllabmcgill/rlcourse-april-7-dy4242407/blob/master/easy21feature.py
        tmp = np.zeros(shape=(3,6,2)) #zeros array of dim 3*6*2
        for i in dealerFeatures(state[1]): # range [0..2] (3 total)
            for j in playerFeatures(state[0]): # range [0..5] (6 total)
                tmp[i,j,action] = 1 # 2 actions total
        x = tmp.flatten()
        
        feats = [i for i, j in enumerate(x) if j == 1]
        return feats
    
    def get_feat_context_for_index(self, feats, c_index):
        context = [0] * 17
    
        index = 0
        for i in range(len(feats)):
            if i == c_index: continue
            context[index] = feats[i]
            index += 1

        return np.array(context)

    def new_episode(self):
        return self.gym.reset()
    
    def end_episode(self):
        self.gym.close()
    
    def step(self, action):
        state, reward, done, _, _ = self.gym.step(action)
        return state, reward, done

class GymWrapper(Env):
    def __init__(self, gym_name):
        self.gym = gym.make(gym_name)
        super().__init__(gym_name, num_actions=self.gym.action_space.n, state_length=-1, num_features=-1)
    
    def new_episode(self):
        return self.gym.reset()
    
    def end_episode(self):
        self.gym.close()
    
    def step(self, action):
        state, reward, done, _, _ = self.gym.step(action)
        return state, reward, done


