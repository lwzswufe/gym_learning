# author='lwz'
# coding:utf-8
import numpy as np


class Board(object):
    def __init__(self, board_size=2):
        self.board_size = board_size
        self.width = board_size * 4 + 1
        self.height = board_size * 4 + 1
        self.bound_1 = board_size * 5
        self.bound_2 = board_size * 3
        self.bound_3 = board_size * 3
        self.bound_4 = board_size * 1
        self.bound_5 = board_size * 3
        self.bound_6 = board_size * 1
        self.state = np.zeros([self.width, self.height])
        for x in range(self.width):
            for y in range(self.height):
                z = self.isin(x, y)
                if z >= 2:
                    self.state[x, y] = 1

    def graphic(self):
        size = 6
        for y in range(self.height):
            print(' '.center(y * 3 + 1), end='')
            for x in range(self.width):
                if self.state[x, y] > 0:
                    print('X'.center(size), end='')
                else:
                    print('.'.center(size), end='')
            print('\r\n')

    def isin(self, x, y):
        z = (self.bound_1 >= x + y >= self.bound_2) +\
            (self.bound_3 >= y >= self.bound_4) +\
            (self.bound_5 >= x >= self.bound_6)
        return z


if __name__ == '__main__':
    b = Board()
    b.graphic()
