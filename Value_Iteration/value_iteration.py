import numpy as np
import tensorflow as tf


def value_iteration(no_states, no_actions, transition_matrix, time_discount, reward_function, no_iteration, value_function = None):
    if value_function == None:
        value_function = np.zeros(( no_states, 1))



    transition_matrix = tf.Variable(transition_matrix, expected_shape= (no_actions, no_states, no_states))



    for i in range(no_iteration):
        value_function = np.array([value_function for i in range(no_actions)])
        value_function = value_update(no_states, no_actions, transition_matrix, time_discount, reward_function, value_function)


    return value_function
def value_update(no_states, no_actions, transition_matrix, time_discount, reward_function, value_function):

    reward = tf.Variable(initial_value=reward_function(), expected_shape=(1, no_states, no_actions))
    value_function =  tf.Variable(initial_value= value_function, dtype=tf.float64)

    value_function = tf.reshape( reward + time_discount*tf.transpose(tf.matmul(transition_matrix, value_function))[0,:,:], (no_states,no_actions,1))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        value_function = np.max(sess.run(value_function), 1)

        return value_function





#EXAMPLE

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



V = value_iteration(25, 4, x, 1, reward, 3)
B = np.array([[V[i][0] for i in range(j, j+5)] for j in range(0,25,5)])
print(B)
