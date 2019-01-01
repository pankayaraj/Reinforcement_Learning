import numpy as np
import tensorflow as tf


class eligibility_trace():

    def __init__(self, no_states, lambda_v, r):

        self.E = np.zeros(shape=(1,no_states))
        self.lambda_v = lambda_v
        self.r = r


    def return_E(self):

        return self.E
    def expand(self):

        temp = np.zeros(shape=(1, 1))
        self.E = np.concatenate((self.E, temp), axis=1)
        self.no_states += 1


    def general_iterate(self):

        self.E = self.r*self.lambda_v*self.E

    def equi_state_iterate(self, state_no):

        self.E[:, state_no] += 1




class State_action():

    def __init__(self, state_action, lambda_v, r):

        #every column is a state

        self.state_action = state_action  # this is the state_action representation of all states and action
        self.no_states = np.shape(state_action)[1]
        self.state_size = np.shape(state_action)[0]

        self.E = eligibility_trace( no_states=self.no_states, lambda_v=lambda_v, r=r)

    def expand(self, new_state_action):

        self.state_action = np.concatenate((self.state_action, new_state_action), axis=1)

# tabular action value function
class Action_value():

    def __init__(self, state_no):
        self.action_value = np.zeros(shape= (1, state_no))

    def get_Q(self, state_no):

        return  self.action_value[:, state_no]

    def update(self, state_no, update, learning_rate):

        self.action_value[:, state_no] += learning_rate*update







def TD_learn_action_value(state_action, action_value, new_state_action_no, curr_state_action_no, R, r, learning_rate):

    delta = R + r*action_value.get_Q(new_state_action_no) - action_value.get_Q(curr_state_action_no)


    #for all states
    state_action.E.equi_state_iterate(curr_state_action_no)

    no_states = state_action.no_states
    state_action.E.general_iterate()  #since it does it for everything in a single call

    for i in range(no_states):

        action_value.update(state_no=i, update=delta*state_action.E.return_E()[:,i], learning_rate= learning_rate )

          #for a state




