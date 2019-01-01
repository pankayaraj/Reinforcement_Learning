import numpy as np
from RL.TD_learning import Action_value

def epsilon_greedy(action_value, next_actions, epsilon):

    next_action_value =  np.array([action_value.get_Q(i)[0] for i in next_actions])




    x = np.random.binomial(n=1, p= epsilon, size=1)

    if x[0] == 1:
        return next_actions[np.argmax(next_action_value)]

    else:
        n = np.random.randint(len(next_actions))
        return next_actions[n]


