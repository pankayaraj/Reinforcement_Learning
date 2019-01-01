from RL.TD_learning import *
import numpy as np
from RL.puzzels import *
from RL.Epsilon_greedy import epsilon_greedy

size  = 10
puzz = Simple(size, 2)

S = np.zeros(shape=(2,size*size))
for i in range(size):
    for j in range(size):
        n = i*10 + j
        S[0][n] = j
        S[1][n] = i


S_A = State_action(state_action=S, lambda_v=0.8, r=0.4)
A_V = Action_value(state_no=100)


for i in range(20):

    start = [0,0]
    x= start[0]
    y= start[1]


    R = 0
    print("episode" + str(i))
    steps = 0
    while R != 1:3
        R = puzz.reward(x, y)
    #for i in range(3):
        steps += 1
        curr_state = puzz.state2action(x,y)
        next_actions = puzz.next_action(x,y)
        new_state = epsilon_greedy(A_V, next_actions, 0.2)

        TD_learn_action_value(S_A, A_V,new_state, curr_state, R, r=0.4, learning_rate=0.5 )

        x = new_state//10
        y = new_state%10



    print(x,y)
    print(steps)

print(puzz.action_into_puzzle_max(A_V))
print(puzz.action_into_puzzle(A_V))
with open('your_file.txt', 'w') as f:
    for item in puzz.action_into_puzzle_max(A_V):
        f.write(" ".join(str(x) for x in item))
        f.write("\n")
#test phase

start = [9,9]
x= start[0]
y= start[1]

R = puzz.reward(start[0], start[1])
print("test")
steps = 0
while R != 1:
    steps += 1
    curr_state = puzz.state2action(x,y)
    next_actions = puzz.next_action(x,y)

    max_a = next_actions[0]
    max_a_value = A_V.get_Q(next_actions[0])

    new_state = epsilon_greedy(A_V, next_actions, 1)
    x = new_state//10
    y = new_state%10
    R = puzz.reward(x, y)
    print(x,y)

print(steps)


'''
TD_learn_action_value(S_A, A_V, 1, 2, 10, 0.4 )
print(A_V.action_value)
TD_learn_action_value(S_A, A_V, 2, 3, 10, 0.4 )
print(A_V.action_value)

TD_learn_action_value(S_A, A_V, 3, 2, 10, 0.4 )
print(A_V.action_value)
'''
