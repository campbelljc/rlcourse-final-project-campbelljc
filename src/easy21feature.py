#ref: https://raw.githubusercontent.com/rllabmcgill/rlcourse-april-7-dy4242407/master/easy21feature.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 23:52:06 2017

@author: yuedong
"""
import numpy as np
#%% code the state-action paris into features

# INPUT
#   dealerState: card value of the dealer, integer between 1 and 10
# OUTPUT
#   boolean vector coding the dealer card to 3 intervals
def dealerFeatures(dealerState):
    assert 1 <= dealerState <= 10
    dealerVec = np.array([dealerState in [1,2,3,4], dealerState in [4,5,6,7], 
                   dealerState in [7,8,9,10]]).astype('int')
    return np.where(dealerVec)[0]
  
    
# INPUT
#   playerState: sum of the player, integer between 1 and 21
# OUTPUT
#   boolean vector coding the player card interval on 6 bits
def playerFeatures(playerState):
    assert 1 <= playerState <= 21
    playerVec = np.array([playerState in [1,2,3,4,5,6], 
                     playerState in [4,5,6,7,8,9],
                     playerState in [7,8,9,10,11,12],
                     playerState in [10,11,12,13,14,15],
                     playerState in [13,14,15,16,17,18],
                     playerState in [16,17,18,19,20,21]]).astype('int')
    return np.where(playerVec)[0]


# INPUTS
#   s: state =(playerState,dealerState) (as defined in env._step)
#   a: action, integer: HIT(1) or STICK(0)
# returns a binary vector of length 36 representing the features
def phi(s, a):
    tmp = np.zeros(shape=(3,6,2)) #zeros array of dim 3*6*2
    #putting one where a feature is on
    for i in dealerFeatures(s[1]):
        for j in playerFeatures(s[0]):
            tmp[i,j,a] = 1 
    return(tmp.flatten()) #returning 'vectorized' (1-dim) array
#%%