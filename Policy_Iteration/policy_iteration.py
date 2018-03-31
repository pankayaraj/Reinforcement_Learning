import tensorflow as tf
import numpy as np

#Commputes and returens the value function after one iiteration
def single_evaluvate(no_states, no_actions, transition_matrix, policy , time_discount, reward_function, value_function):

    value_function = tf.Variable(initial_value= value_function, dtype=tf.float64)
    reward = tf.Variable(initial_value=reward_function(), expected_shape=(1, no_states, no_actions))

    value_function = tf.matmul(policy,
                               tf.reshape( reward + time_discount*tf.transpose(tf.matmul(transition_matrix, value_function))[0,:,:], (no_states,no_actions,1))
                               )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        V = sess.run(value_function)

    return V
#does the calue function iteration for several times and in the end returns the action value function for ppolicy improvement
def policy_evaluation(no_states, no_actions, transition_matrix, initail_policy , time_discount, reward_function, no_iteration):
    value_function_init = np.zeros(( no_states, 1))
    value_function_init = np.array([value_function_init for i in range(no_actions)])

    policy = tf.Variable(initial_value=initail_policy, expected_shape=(no_states, no_actions, 1), dtype=tf.float64)


    for _ in range(no_iteration):
        value_function = single_evaluvate(no_states, no_actions, transition_matrix, policy , time_discount, reward_function, value_function_init)
        value_function_init = np.array([value_function[:,:,0] for i in range(no_actions)])

#Upto now the iteration was made on Value function now only we r gonna compute the action value function
    value_function_temp = tf.Variable(initial_value= value_function_init, dtype=tf.float64)
    reward = tf.Variable(initial_value=reward_function(), expected_shape=(1, no_states, no_actions))
    acion_value_function = tf.reshape( reward + time_discount*tf.transpose(tf.matmul(transition_matrix, value_function_temp))[0,:,:], (no_states,no_actions,1))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(acion_value_function)


#Estimate the MDP for a certian iterations and then improves the policy once
def policy_iterate(no_states, no_actions, transition_matrix, policy , time_discount, reward_function, no_iteration):

    transition_matrix = tf.Variable(transition_matrix, expected_shape= (no_actions, no_states, no_states))
    action_value_function = policy_evaluation(no_states, no_actions, transition_matrix, policy , time_discount, reward_function, no_iteration)


    for i in range(no_states):
        if(np.max(policy[i, :, :]) != 0):
            avf = action_value_function[i, :, :]
            x = [0 for _ in range(no_actions)]
            x[np.argmax(avf)] = 1
            policy[i, :, :] = x
    return (policy)

#EXAMPLE : SMALL GRID WORLD WITH END STATES 0, 24 th block
def reward():
    r = np.zeros((1, 25, 4)) - 1
    r[0,0, :] = [0,0,0,0]
    r[0,24, :] = [0,0,0,0]

    return r


x = np.array([np.zeros((25, 25)), np.zeros((25, 25)), np.zeros((25, 25)), np.zeros((25, 25)) ])

# 0 right
for z in range(25):
    l = 0
    if z%5 + 1 < 5:
        x[0][z][z+1] = 1
    else:
        x[0][z][z] = 1

# 1 left
for z in range(25):
    l = 0
    if z%5 - 1 >= 0:
        x[1][z][z-1] = 1
    else:
        x[1][z][z] = 1

# 2 down
for z in range(25):
    if z//5 + 1 < 5:
        x[2][z][z+5] = 1
    else:
        x[2][z][z] = 1

# 3 up
for z in range(25):
    if z//5 - 1 >= 0:
        x[3][z][z-5] = 1
    else:
        x[3][z][z] = 1

policy = np.zeros((25, 1, 4)) + 0.25
policy[0,:,:] = [0,0,0,0]
policy[24,:,:] = [0,0,0,0]
'''
V = policy_evaluation(25, 4, x, policy, 1, reward, 10)
print(V)
V = V[:,:,0]
B = np.array([[V[i][0] for i in range(j, j+5)] for j in range(0,25,5)])
print(B)
'''
for i in range(2):
    policy = policy_iterate(25, 4, x, policy, 1, reward, 10)
    P = policy[:,0,:]
    B = np.array([[np.argmax(P[i]) for i in range(j, j+5)] for j in range(0,25,5)])

print(B)
