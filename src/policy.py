# from keras-rl
import numpy as np

class Policy(object):
    def _set_agent(self, agent):
        self.agent = agent

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        return {}

class GreedyQPolicy(Policy):
    def __init__(self):
        super(GreedyQPolicy, self).__init__()

    def select_action(self, q_values):
        #assert q_values.ndim == 1
        return np.argmax(q_values)

class GreedyPossibleQPolicy(Policy):
    def __init__(self):
        super(GreedyPossibleQPolicy, self).__init__()

    def select_action(self, q_values):
        #assert q_values.ndim == 1
        action_validities = self.agent.env.get_action_validities()
        assert any(act is True for act in action_validities)
        
        action = np.argmax([qv if action_validities[i] else -np.inf for i, qv in enumerate(q_values)])
        assert action_validities[action]
        return action

class LinearAnnealedPolicy(Policy):
    def __init__(self, inner_policy, attr, value_max, value_min, value_test, nb_steps):
        if not hasattr(inner_policy, attr):
            raise ValueError('Policy "{}" does not have attribute "{}".'.format(attr))

        super(LinearAnnealedPolicy, self).__init__()

        self.inner_policy = inner_policy
        self.attr = attr
        self.value_max = value_max
        self.value_min = value_min
        self.value_test = value_test
        self.nb_steps = nb_steps
    
    def _set_agent(self, agent):
        self.agent = agent
        self.inner_policy._set_agent(agent)
    
    def get_current_value(self):
        if self.agent.training:
            # Linear annealed: f(x) = ax + b.
            a = -float(self.value_max - self.value_min) / float(self.nb_steps)
            b = float(self.value_max)
            value = max(self.value_min, a * float(self.agent.cur_action) + b)
        else:
            value = self.value_test
        return value

    def select_action(self, **kwargs):
        setattr(self.inner_policy, self.attr, self.get_current_value())
        return self.inner_policy.select_action(**kwargs)

    @property
    def metrics_names(self):
        return ['mean_{}'.format(self.attr)]

    @property
    def metrics(self):
        return [getattr(self.inner_policy, self.attr)]

    def get_config(self):
        config = super(LinearAnnealedPolicy, self).get_config()
        config['attr'] = self.attr
        config['value_max'] = self.value_max
        config['value_min'] = self.value_min
        config['value_test'] = self.value_test
        config['nb_steps'] = self.nb_steps
        config['inner_policy'] = get_object_config(self.inner_policy)
        return config

class EpsGreedyQPolicy(Policy):
    def __init__(self, eps=.1):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        #assert q_values.ndim == 1
        nb_actions = len(q_values) #.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.random_integers(0, nb_actions-1)
        else:
            action = np.argmax(q_values)
        return action
    
    def get_config(self):
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config

class EpsGreedyPossibleQPolicy(Policy):
    def __init__(self, eps=.1):
        super(EpsGreedyPossibleQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        #assert q_values.ndim == 1
        nb_actions = len(q_values) #q_values.shape[0]
        
        action_validities = self.agent.env.get_action_validities()
        if not any(act is True for act in action_validities):
            print(self.agent.env.state)
            assert False
        
        if np.random.uniform() < self.eps:
            possible_actions = [i for i in range(nb_actions) if action_validities[i]]
            action = np.random.choice(possible_actions)
            #action = np.random.random_integers(0, nb_actions-1)
        else:
            action = np.argmax([qv if action_validities[i] else -np.inf for i, qv in enumerate(q_values)])
        assert action_validities[action]
        return action

    def get_config(self):
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config