import cts.model as model

import numpy as np
from math import exp

class SimpleDensityModel(object):
    def __init__(self, frame_shape):
        self.convolutional_model = model.CTS(context_length=0)
        self.frame_shape = frame_shape
        
    def update(self, frame):
        assert(frame.shape == self.frame_shape)
        
        total_log_probability = 0.0
        # We simply apply the CTS update to each pixel.
        for y in range(frame.shape[0]):
            #for x in range(frame.shape[1]):
            # Convert all 3 channels to an atomic colour.
            #colour = rgb_to_symbol(frame[y, x])
            total_log_probability += self.convolutional_model.update(context=[], symbol=frame[y])

        return total_log_probability
    
    def log_prob(self, frame):
        assert(frame.shape == self.frame_shape)
        
        total_log_probability = 0.0
        # We simply apply the CTS update to each pixel.
        for y in range(frame.shape[0]):
            #for x in range(frame.shape[1]):
            # Convert all 3 channels to an atomic colour.
            #colour = rgb_to_symbol(frame[y, x])
            total_log_probability += self.convolutional_model.log_prob(context=[], symbol=frame[y])

        return total_log_probability
    
    def sample(self):
        output_frame = np.zeros(self.frame_shape, dtype=np.uint8)
        
        for y in range(output_frame.shape[0]):
            for x in range(output_frame.shape[1]):
                # Use rejection sampling to avoid generating non-Atari colours.
                #colour = self.convolutional_model.sample(context=[], rejection_sampling=True)
                output_frame[y, x] = self.convolutional_model.sample(context=[], rejection_sampling=True) #symbol_to_rgb(colour)
        
        return output_frame

class LocationDependentDensityModel(object):
    def __init__(self, state_length, context_functor=None, state_splits=None, alphabet=None):
        self.state_length = state_length
        if state_splits is None:
            assert context_functor is not None
            
            context_length = state_length - 1
            self.models = [None] * state_length
            self.model_ctxt_fns = [None] * state_length
            print("ctxt len:", context_length)
        
            for y in range(state_length):
                self.models[y] = model.CTS(context_length=context_length, alphabet=alphabet)
                self.model_ctxt_fns[y] = context_functor
            self.ranges = None
        
        else:
            self.models = []
            self.model_ctxt_fns = []
            
            for state_split_range in state_splits:
                context_length, context_fn = state_splits[state_split_range]
                self.models.append(model.CTS(context_length=context_length))
                self.model_ctxt_fns.append(context_fn)
            self.ranges = state_splits.keys()
    
    def update(self, frame):
        total_log_probability = 0.0
        if self.ranges is None:
            for y in range(self.state_length):
                context = self.model_ctxt_fns[y](frame, y)
                total_log_probability += self.models[y].update(context=context, symbol=frame[y])
        else:
            for i, range_ in enumerate(self.ranges):
                #if self.models[i].context_length == 0 and len(range_) > 1:
                #    # then encode this particular range into one symbol..
                #    symbol = 0
                #    for j, index in enumerate(range_):
                #        symbol += frame[index] * (j + 1)
                #    total_log_probability += self.models[i].update(context=[], symbol=symbol)                    
                #else:
                #for index in range_:
                context = self.model_ctxt_fns[i](frame, i)
                total_log_probability += self.models[i].update(context=context, symbol=frame[i])
        
        return exp(total_log_probability)
    
    def log_prob(self, frame):        
        total_log_probability = 0.0
        if self.ranges is None:
            for y in range(self.state_length):
                context = self.model_ctxt_fns[y](frame, y)
                total_log_probability += self.models[y].update(context=context, symbol=frame[y])
        else:
            for i, range_ in enumerate(self.ranges):
                context = self.model_ctxt_fns[i](frame, i)
                total_log_probability += self.models[i].update(context=context, symbol=frame[i])
                
                #if self.models[i].context_length == 0 and len(range_) > 1:
                #    # then encode this particular range into one symbol..
                #    symbol = 0
                #    for j, index in enumerate(range_):
                #        symbol += frame[index] * (j + 1)
                #    total_log_probability += self.models[i].log_prob(context=[], symbol=symbol)                    
                #else:
                #    for index in range_:
                #        context = self.model_ctxt_fns[i](frame, index)
                #        total_log_probability += self.models[i].log_prob(context=context, symbol=frame[index])
        
        return exp(total_log_probability)
    