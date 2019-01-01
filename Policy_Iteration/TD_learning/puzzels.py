import numpy  as np


class Simple():

    def __init__(self, size, end_points):
        self.size = size
        self.P = np.array([[0 for _ in range(size)] for __ in range(size)])

        '''
        for i in range(end_points):
            x  = np.random.randint(size)
            y = np.random.randint(size)

            self.P[x][y] = 1
        '''

        self.P[5][8] = 1
        self.P[7][2] = 1
    def reward(self, x, y):
        if self.P[x][y] == 0:
            return 0
        else:
            return 1


    def action_into_puzzle(self, action_value):

        A = np.array([[0.0 for _ in range(self.size)] for __ in range(self.size)])
        for i in range(self.size):
            for j in range(self.size):

                A[i][j] = action_value.get_Q(i*10 + j)

        return A

    def action_into_puzzle_max(self, action_value):


        B = np.array([[0 for _ in range(self.size)] for __ in range(self.size)])

        max_a = 0
        for i in range(self.size):
            for j in range(self.size):
                n = self.next_action(i, j)

                n_a = np.array([action_value.get_Q(k)[0] for k in n])

                B[i][j] = n[np.argmax(n_a)]


        return B

    def state2action(self, x, y):
         return x*10 + y

    def next_action(self, x, y):
        A = []
        if x != 0 and x != 9 and y != 0 and y != 9:
            A.append(x*10 + y + 1)
            A.append(x*10 + y - 1)
            A.append((x-1)*10 + y )
            A.append((x+1)*10 + y )

        elif x == 0 and y == 0:
            A.append(1)
            A.append(10)

        elif x ==0 and y == 9:

            A.append(8)
            A.append(19)

        elif x == 9 and y == 0:

            A.append(80)
            A.append(91)

        elif x == 9 and y == 9:
            A.append(98)
            A.append(89)

        elif x == 0:
            A.append(x*10 + y + 1)
            A.append(x*10 + y - 1)
            A.append((x+1)*10 + y )
        elif x == 9:
            A.append(x*10 + y + 1)
            A.append(x*10 + y - 1)
            A.append((x-1)*10 + y )

        elif y == 0:
            A.append(x*10 + y + 1)
            A.append((x-1)*10 + y )
            A.append((x+1)*10 + y )

        elif y == 9:
            A.append(x*10 + y - 1)
            A.append((x-1)*10 + y )
            A.append((x+1)*10 + y )

        return A

