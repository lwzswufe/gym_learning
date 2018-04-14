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
        self.player_num = 2
        self.players_net = [[], []]
        self.players_pegs = []
        self.step_n = 0
        self.step_max = self.board_size * self.pegs_num * 4
        self.availables = []
        self.point_num = size * size + 1
        self.state = np.zeros(self.point_num, dtype=np.int)

        self.walk = np.zeros([self.point_num, len(self.basic_actions)], dtype=np.int)  # 坐标转移矩阵 走
        self.jump = np.zeros([self.point_num, len(self.basic_actions)], dtype=np.int)  # 坐标转移矩阵 跳
        self.walk_mat_init()
        self.jump_mat_init()
        self.jump_extend_points = dict()
        self.max_score = 0
        self.get_max_score()
        self.current_player_id = 0
        self.last_move = None

    def get_max_score(self):
        '''
        获取最大得分
        :return:
        '''
        score = 0
        for i in range(self.pegs_size):
            score += (2 * (self.board_size - 1) - i) * (i + 1)
        self.max_score = score
                        
    def walk_mat_init(self):
        '''
        初始化单步移动状态转移矩阵
        :return:
        '''
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
        '''
        初始化单步跳跃状态转移矩阵
        :return:
        '''
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
        board 初始化board
        :return:
        '''
        self.last_move = None
        self.step_n = 0
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

    def graphic(self):
        '''
        绘图程序
        :return:
        '''
        size = 6
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

    def get_availables(self):
        '''
        获取玩家的所有可行动作
        :param player_id:
        :return:
        '''
        actions = []
        self.get_continuously_jump()
        for i in range(self.pegs_num):
            next_locs = self.get_available(pegs_id=i)
            actions += list(zip([i] * len(next_locs), next_locs))
        return actions

    def get_available(self, pegs_id):
        '''
        获取单一棋子的所有可行动作
        :return:
        '''
        player_id = self.current_player_id
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
        '''
        获取可连续跳跃的空白点
        :return:
        '''
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
        '''
        获取可连续跳跃的等价空白点
        :return:
        '''
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

    def get_score(self):
        '''
        获取当前玩家评分
        :return:
        '''
        score = [0, 0]
        for loc in self.players_pegs[0, :]:
            (x, y) = self.get_location(loc)
            score[0] += x + y

        for loc in self.players_pegs[1, :]:
            (x, y) = self.get_location(loc)
            score[1] += (self.board_size - 1) * 2 - x - y
        return score

    def game_end(self):
        '''
        判断游戏是否结束
        :return: 游戏结束 胜利者
        '''
        score = self.get_score()
        # print(score)
        if max(score) >= self.max_score:
            winner = np.argmax(score)
            return True, winner
        elif self.step_n > self.step_max:
            winner = np.argmax(score)
            return True, winner
        else:
            return False, -1

    def do_move(self, action):
        '''
        移动棋子
        :param player_id:
        :param pegs_id:
        :param target_loc:
        :return:
        '''
        (pegs_id, target_loc) = action
        player_id = self.current_player_id
        loc_old = self.players_pegs[player_id, pegs_id]
        self.players_pegs[player_id, pegs_id] = target_loc
        self.state[loc_old] = -1
        self.state[target_loc] = player_id

        self.step_n += 1
        self.current_player_id = self.step_n % self.player_num
        self.last_move = action

    def update_state(self):
        state = - np.ones(self.point_num, dtype=np.int)
        for player_id in range(self.player_num):
            for peg_id in range(self.pegs_num):
                loc = self.players_pegs[player_id, peg_id]
                state[loc] = player_id
        state[-1] = -2
        self.state = state

    def get_location(self, loc_id=7):
        '''
        位置标号转化为位置坐标(x, y)
        :param loc_id:
        :return:
        '''
        x = int(loc_id % self.board_size)
        y = int(loc_id // self.board_size)
        return (x, y)

    def get_location_id(self, loc=(0, 2)):
        '''
        位置坐标(x, y)转化为位置编号
        :param loc:
        :return:
        '''
        return int(loc[0] + loc[1] * self.board_size)

    def get_current_state(self):
        '''
        返回四个矩阵
        0： 当前player 棋子的位置
        1： 对手player 棋子的位置
        2： 对手最近一次落子的位置
        3： 当前player是不是先手
        :return:
        '''
        player_id = self.current_player_id
        square_state = np.zeros([4, self.board_size, self.board_size])

        square_state[0][self.players_pegs[player_id, :] // self.board_size,
                        self.players_pegs[player_id, :] % self.board_size] = 1.0

        square_state[1][self.players_pegs[1 - player_id, :] // self.board_size,
                        self.players_pegs[1 - player_id, :] % self.board_size] = 1.0

        # indicate the last move location
        if self.last_move is not None:
            square_state[2][self.get_location(self.last_move[1])] = 1.0

        if self.current_player_id == 0:
            square_state[3][:, :] = 1.0  # 先手标记

        return square_state

    def get_probs_map(self, actions, probs):
        start_probs = np.zeros([self.board_size, self.board_size])
        end_probs = np.zeros([self.board_size, self.board_size])
        for i in range(len(actions)):
            peg_id, target_loc = actions[i]
            x, y = self.get_location(target_loc)
            end_probs[x, y] = max(end_probs[x, y], probs[i])
            x, y = self.get_location(self.players_pegs[self.current_player_id, peg_id])
            start_probs[x, y] = max(start_probs[x, y], probs[i])

        return start_probs, end_probs


def list_to_net(size, pegs_list):
    net = np.zeros(size, size)
    for (x, y) in pegs_list:
        net[x, y] = 1
    return net


class Board_new(Board):
    def random_strategy(self, player_id=0, actions=[]):
        (peg_id, target_loc) = random.sample(actions, 1)[0]
        return peg_id, target_loc

    def step(self, player_id=0):
        actions = self.get_availables()
        action = self.greed_strategy(player_id, actions)
        self.do_move(action)
        # self.graphic()
        time.sleep(0.5)

    def self_play(self):
        flag = 0
        while True:
            self.step(flag % 2)
            end, winner = self.game_end()
            if end:
                print('game end the winner is: {}'.format(winner))
                break
            flag += 1
        play_data = self.get_current_state()
        return play_data

    def action_evaluate(self, player_id, peg_id, target_loc):
        peg_loc = self.players_pegs[player_id, peg_id]
        (x, y) = self.get_location(peg_loc)
        (x_, y_) = self.get_location(target_loc)
        if player_id == 0:
            return (x_ + y_) - (x + y)
        else:
            return (x + y) - (x_ + y_)

    def greed_strategy(self, player_id, actions):
        scores = list()
        for (peg_id, target_loc) in actions:
            score = self.action_evaluate(player_id, peg_id, target_loc)
            scores.append(score)

        max_score = max(scores)
        actions_best = [actions[i] for i in range(len(scores)) if scores[i] >= max_score]
        (peg_id, target_loc) = random.sample(actions_best, 1)[0]
        return peg_id, target_loc

    def play_with_computer(self):
        pass


if __name__ == "__main__":
    b = Board_new()
    b.init_board()
    data = b.self_play()


