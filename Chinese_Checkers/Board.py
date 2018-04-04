# author='lwz'
# coding:utf-8

import numpy as np
import random
import itertools
import time


class Board(object):
    def __init__(self, size=6, pegs_size=3):
        self.board_size = size
        self.pegs_num = int(pegs_size * (pegs_size + 1) / 2)
        self.pegs_size = pegs_size
        # self.wall_states = np.array()
        self.terminate_states = 0
        self.basic_actions = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]]
        #          #     右      左        下        上      右上     左下
        # self.net = np.zeros([self.board_size, self.board_size])
        self.players_net = [[], []]
        self.players_pegs = []
        self.step_n = 0
        self.availables = []
        self.point_num = size * size + 1
        self.state = np.zeros(self.point_num, dtype=np.int)

        self.walk = np.zeros([self.point_num, len(self.basic_actions)], dtype=np.int)  # 坐标转移矩阵 走
        self.jump = np.zeros([self.point_num, len(self.basic_actions)], dtype=np.int)  # 坐标转移矩阵 跳
        self.walk_mat_init()
        self.jump_mat_init()
        self.jump_extend_points = dict()
                        
    def walk_mat_init(self):
        for x in range(self.board_size):
            for y in range(self.board_size):
                flag = y * self.board_size + x
                for i, action in enumerate(self.basic_actions):
                    x_ = x + action[0]
                    y_ = y + action[1]
                    if max(x_, y_) <= self.board_size - 1 and min(x_, y_) >= 0:
                        self.walk[flag, i] = y_ * self.board_size + x_
                    else:
                        self.walk[flag, i] = -1
        self.walk[-1, :] = -1
                        
    def jump_mat_init(self):
        for x in range(self.board_size):
            for y in range(self.board_size):
                flag = y * self.board_size + x
                for i, action in enumerate(self.basic_actions):
                    x_ = x + action[0] * 2
                    y_ = y + action[1] * 2
                    if max(x_, y_) <= self.board_size - 1 and min(x_, y_) >= 0:
                        self.jump[flag, i] = y_ * self.board_size + x_
                    else:
                        self.jump[flag, i] = -1
        self.jump[-1, :] = -1

    def init_board(self):
        '''
        board 初始化
        :return:
        '''
        self.players_pegs = np.zeros([2, self.pegs_num], dtype=np.int)

        pegs_list = itertools.product(range(self.pegs_size), range(self.pegs_size))
        pegs_list = [(x, y) for (x, y) in pegs_list if x + y < self.pegs_size]
        self.players_pegs[0, :] = np.array([self.get_location_id(loc) for loc in pegs_list])

        pegs_list = [(self.board_size - 1 - x, self.board_size - 1 - y)
                                for (x, y) in pegs_list]
        self.players_pegs[1, :] = np.array([self.get_location_id(loc) for loc in pegs_list])

        self.state = -1 * np.ones(self.board_size * self.board_size + 1, dtype=np.int)
        self.state[-1] = -2
        for i, player in enumerate(self.players_pegs):
            for peg in player:
                self.state[peg] = i
        # self.players_net[0] = list_to_net(self.players_pegs[0])
        # self.players_net[1] = list_to_net(self.players_pegs[1])

        # 可行策略初始化 策略树展开必备条件
    @property
    def pegs(self):
        return self.players_pegs[0] + self.players_pegs[1]

    @property
    def net(self):
        return list_to_net(self.board_size, self.pegs)

    def graphic(self):
        size = 4
        for y in range(self.board_size):
            print(' '.center(y * 2 + 1), end='')
            for x in range(self.board_size):
                loc = self.get_location_id((x, y))
                if loc in self.players_pegs[0, :]:
                    print('X'.center(size), end='')
                elif loc in self.players_pegs[1, :]:
                    print('O'.center(size), end='')
                else:
                    print('.'.center(size), end='')
            print('\r\n')

    def get_availables(self, player_id):
        actions = []
        self.get_continuously_jump()
        for i in range(self.pegs_num):
            next_locs = self.get_available(player_id=player_id, pegs_id=i)
            actions += list(zip([i] * len(next_locs), next_locs))
        return actions

    def get_available(self, player_id, pegs_id):
        '''
        获取可行状态
        :return:
        '''
        loc = self.players_pegs[player_id][pegs_id]

        walk = self.walk[loc, :][self.state[self.walk[loc]] == -1]
        jump = self.jump[loc, :][(self.state[self.jump[loc]] == -1) & (self.state[self.walk[loc, :]] > -1)]

        if len(self.jump_extend_points) > 0 and len(jump) > 0:
            jumps = list()
            for point in jump:
                if point in self.jump_extend_points.keys():
                    jumps += self.jump_extend_points[point]
            jump = list(set(jumps + list(jump)))

        return list(walk) + list(jump)      # 跳跃点 移动点不会重合

    def get_continuously_jump(self):
        points = np.array(range(self.point_num), dtype=np.int)
        st_point = []
        ed_point = []
        for action in [0, 2, 4]:
            j = points[(self.state == -1) & (self.state[self.walk[:, action]] > -1)
                       & (self.state[self.jump[:, action]] == -1)]
            st_point += list(j)
            ed_point += list(self.jump[j, action])

        self.jump_extend(st_point, ed_point)

    def jump_extend(self, st_point, ed_point):
        key = set(st_point + ed_point)
        if len(key) == len(st_point) + len(ed_point):
            jumps = dict()
            for (st, ed) in zip(st_point, ed_point):
                jumps[st] = [ed]
                jumps[ed] = [st]
            self.jump_extend_points = jumps
            return

        value = [-1] * len(key)
        set_links = dict(zip(key, value))
        point_set = list()
        for (st, ed) in zip(st_point, ed_point):
            link_st = set_links[st]
            link_ed = set_links[ed]
            if max(link_st, link_ed) < 0:      # 均未写入
                point_set.append([st, ed])
                set_links[st] = len(point_set) - 1
                set_links[ed] = len(point_set) - 1
            elif min(link_st, link_ed) >= 0:   # 均写入
                if link_st != link_ed:         # 合并集合
                    point_set[link_st] += point_set[link_ed]
                    for point in point_set[link_ed]:
                        set_links[point] = link_st
            elif link_ed >= 0:
                set_links[st] = link_ed
                point_set[link_ed].append(st)
            else:
                set_links[ed] = link_st
                point_set[link_st].append(ed)

        jumps = dict()
        for point in key:
            jumps[point] = point_set[set_links[point]]

        self.jump_extend_points = jumps

    def game_end(self):
        if self.state in self.terminate_states:
            return True, 10.0 / (self.step_n + 1)
        elif self.step_n > self.step_max:
            return True, -1
        else:
            return False, 10.0 / (self.step_n + 1)

    def do_move(self, player_id, pegs_id, target_loc):
        loc_old = self.players_pegs[player_id, pegs_id]
        self.players_pegs[player_id, pegs_id] = target_loc
        self.state[loc_old] = -1
        self.state[target_loc] = player_id

        self.step_n += 1

    def get_location(self, loc_id=7):
        x = int(loc_id % self.board_size)
        y = int(loc_id // self.board_size)
        return (x, y)

    def get_location_id(self, loc=(0, 2)):
        return int(loc[0] + loc[1] * self.board_size)


def list_to_net(size, pegs_list):
    net = np.zeros(size, size)
    for (x, y) in pegs_list:
        net[x, y] = 1
    return net


class Board_new(Board):
    def random_move(self, player_id=0):
        actions = self.get_availables(player_id)
        (peg_id, target_loc) = random.sample(actions, 1)[0]
        self.do_move(player_id, peg_id, target_loc)
        self.graphic()
        time.sleep(0.5)

    def self_play(self):
        flag = 0
        while flag < 30:
            self.random_move(0)
            self.random_move(1)
            flag += 1

    def play_with_computer(self):
        pass


if __name__ == "__main__":
    b = Board_new()
    b.init_board()
    b.self_play()


