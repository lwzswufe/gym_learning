# author='lwz'
# coding:utf-8

import numpy as np
import random
from MCTS import MCTSPlayer as Player


class Board(object):
    def __init__(self, cell_rows=7, cell_cols=7, barrier_ratio=0.2):
        self.cell_cols = cell_cols
        self.cell_rows = cell_rows
        self.cell_num = cell_cols * cell_rows
        self.step_max = 2 * (cell_rows + cell_cols)
        # self.wall_states = np.array()
        self.terminate_states = 0
        self.actions = [0, 1, 2, 3]
        #          # n上0  e右1  s下2  w左3
        self.barrier_ratio = barrier_ratio
        self.wall_states = random.sample(range(self.cell_num), 1 + int(self.cell_num * self.barrier_ratio))
        self.blank_state = [state for state in range(self.cell_num) if state not in self.wall_states]
        self.terminate_states = [self.wall_states.pop(-1)]
        self.width = cell_cols
        self.height = cell_rows
        self.step_n = 0
        self.availables = []

    def init_board(self):
        '''
        board 初始化
        :return:
        '''
        self.state = random.sample(self.blank_state, 1)[0]
        self.t = np.zeros([self.cell_num, 4])  # 状态转移矩阵
        self.step_n = 0

        for i in range(self.cell_num):
            if i + self.cell_cols in self.wall_states:
                self.t[i, 0] = i
            elif i + self.cell_cols >= self.cell_num:
                self.t[i, 0] = i
            else:
                self.t[i, 0] = i + self.cell_cols

            if i + 1 in self.wall_states:
                self.t[i, 1] = i
            elif (i + 1) % self.cell_cols == 0:
                self.t[i, 1] = i
            else:
                self.t[i, 1] = i + 1

            if i - self.cell_cols in self.wall_states:
                self.t[i, 2] = i
            elif i - self.cell_cols < 0:
                self.t[i, 2] = i
            else:
                self.t[i, 2] = i - self.cell_cols

            if i - 1 in self.wall_states:
                self.t[i, 3] = i
            elif i % self.cell_cols == 0:
                self.t[i, 3] = i
            else:
                self.t[i, 3] = i - 1

        self.availables = self.get_availables()
        # 可行策略初始化 策略树展开必备条件
        
    def get_availables(self):
        '''
        获取可行状态
        :return:
        '''
        turn_over = self.t[self.state, :]
        arr = np.array(self.actions)[turn_over != self.state]
        return arr

    def get_state_id(self, row_id, col_id):
        '''
        坐标转化为位置
        :return:
        '''
        state_id = (row_id - 1) * self.cell_cols + col_id
        return state_id

    def game_end(self):
        if self.state in self.terminate_states:
            return True, 10.0/(self.step_n+1)
        elif self.step_n > self.step_max:
            return True, -1
        else:
            return False, 10.0/(self.step_n+1)

    def do_move(self, action):
        self.state = int(self.t[self.state, int(action)])
        self.step_n += 1
        self.availables = self.get_availables()


class Game(object):
    """game server"""

    def __init__(self, board=Board(), player=Player(),  **kwargs):
        self.board = board
        self.player = player

    def graphic(self, board):
        """Draw the board and show game info"""
        width = board.width
        height = board.height
        size = 4

        # print("Player", player1, "with X".rjust(3))
        # print("Player", player2, "with O".rjust(3))

        for x in range(width):
            print("{0:4}".format(x), end='')
        print('\r\n')

        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j

                if loc in board.wall_states:
                    print('X'.center(size), end='')
                elif loc in board.terminate_states:
                    print('O'.center(size), end='')
                elif loc == board.state:
                    print('o'.center(size), end='')
                else:
                    print('_'.center(size), end='')

            print('\r\n')

    def start_play(self, is_shown=1):
        """start a game between two players"""
        self.board.init_board()

        if is_shown:
            self.graphic(self.board)

        while True:
            move = self.player.get_action(self.board)
            # get_action(self.board) 模拟 探索
            self.board.do_move(move)

            if is_shown:
                self.graphic(self.board)

            end, step_n = self.board.game_end()
            if end:
                if is_shown:
                    print("Game end. step:", step_n)
                return step_n

    def start_self_play(self, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        states, mcts_probs = [], []
        while True:
            move, move_probs = self.player.get_action(self.board)

            # store the data
            states.append(self.board.state)
            mcts_probs.append(move_probs)

            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board)
                print('1')

            end, step_n = self.board.game_end()
            if end:
                return step_n, zip(states, mcts_probs, step_n)
