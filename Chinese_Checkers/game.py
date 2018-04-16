# author='lwz'
# coding:utf-8
'''
模拟游戏程序
贪婪策略对阵 MinimaxTree height=2 explore_num=3 胜率为54%
贪婪策略对阵 MinimaxTree height=4 explore_num=3 胜率为48%
贪婪策略对阵 MinimaxTree height=2 explore_num=5 胜率为33%
贪婪策略对阵 MinimaxTree height=2 explore_num=10 胜率为16%
贪婪策略对阵 MinimaxTree height=4 explore_num=5 胜率为14%
'''
from Chinese_Checkers.Board import Board
from Chinese_Checkers.MCTS import MCTS, MCTSPlayer
from Chinese_Checkers.Player import AI_Player, Random_Strategy, Greedy_Strategy
from Chinese_Checkers.MinimaxTree import MiniMaxTree, policy_value_fn_0, policy_value_fn_1
import time
import numpy as np
import random


class Game(object):
    def __init__(self, board):
        self.board = board

    def self_play(self, players, is_shown=False, start_player=None):
        step = 0
        used_time = []
        self.board.init_board()

        if start_player is None:
            rnd = round(random.random())

        for i, player in enumerate(players):
            player.player_id = (i + rnd) % 2
            used_time.append(0)

        while True:
            time_st = time.time()
            player_id = (step + rnd) % len(players)

            # board.update_state()
            action = players[player_id].get_action(self.board)
            self.board.do_move(action)
            if is_shown:
                self.board.graphic()
            end, winner = self.board.game_end()
            if end:
                print('game end the winner is: {} AI_name:{}'.format(winner, players[player_id].name))
                break
            step += 1
            time_ed = time.time()
            used_time[player_id] += time_ed - time_st

        print('usedtime: {:.4f}s  {:.4f}s'.format(used_time[0], used_time[1]))
        return winner, step

    def get_self_play_data(self, players, is_shown=False):
        step = 0
        used_time = []
        self.board.init_board()
        rnd = round(random.random())
        for i, player in enumerate(players):
            player.player_id = (i + rnd) % 2
            player.reset()
            used_time.append(0)

        states, probs_map, current_players = [], [], []
        while True:
            time_st = time.time()
            player_id = (step + rnd) % len(players)

            # board.update_state()
            actions, probs = players[player_id].get_probs(self.board)
            start_probs, end_probs = self.board.get_probs_map(actions, probs)
            action = players[player_id].get_action(self.board, actions, probs)

            states.append(self.board.get_current_state())
            probs_map.append((start_probs, end_probs))
            current_players.append(self.board.current_player_id)

            self.board.do_move(action)
            if is_shown:
                self.board.graphic()
            end, winner = self.board.game_end()
            if end:
                # print('game end the winner is: {} AI_name:{}'.format(winner, players[player_id].name))
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, probs_map, winners_z)


def self_play(board, players, is_shown=False):
    step = 0
    used_time = []
    board.init_board()
    rnd = round(random.random())
    for i, player in enumerate(players):
        player.player_id = (i + rnd) % 2
        used_time.append(0)

    while True:
        time_st = time.time()
        player_id = (step + rnd) % 2
        # board.players_pegs = np.array([
        #    [35, 29, 23, 34, 27, 33],
        #    [0, 6, 15, 1, 7, 2]], dtype=np.int)
        # board.update_state()
        action = players[player_id].get_action(board)
        board.do_move(action)
        if is_shown:
            board.graphic()
        end, winner = board.game_end()
        if end:
            winner = (winner + rnd) % 2
            print('game end the winner is: {} AI_name:{}'.format(winner, players[player_id].name))
            break
        step += 1
        time_ed = time.time()
        used_time[player_id] += time_ed - time_st

    print('usedtime: {:.4f}s  {:.4f}s'.format(used_time[0], used_time[1]))
    return winner, step


def main():
    board = Board()

    # r_s = Random_Strategy()
    # player_0 = AI_Player(r_s)

    player_0 = MCTSPlayer(c_puct=50, n_playout=100)
    # player_0 = MiniMaxTree(height=2, policy_fun=policy_value_fn_1, explore_num=3)
    # player_0 = MiniMaxTree(height=2)

    player_1 = Greedy_Strategy
    # player_1 = MCTSPlayer(c_puct=30, n_playout=100)
    # player_1 = MiniMaxTree(height=4)
    self_play(board, [player_0, player_1], True)


def play_repeat(repeat_time):
    # player_0 = MiniMaxTree(height=2, policy_fun=policy_value_fn_1, explore_num=5)
    # player_0 = MCTSPlayer(c_puct=5, n_playout=100)
    player_0 = MiniMaxTree(height=2, explore_num=3, policy_fun=policy_value_fn_1)

    player_1 = Greedy_Strategy()
    # player_1 = MiniMaxTree(height=4, explore_num=5, policy_fun=policy_value_fn_1)
    # player_1 = MCTSPlayer(c_puct=50, n_playout=100)
    wins = 0
    step_mean = 0
    board = Board()
    for i in range(repeat_time):
        winner, step = self_play(board, [player_0, player_1], False)
        wins += winner

    step_mean /= repeat_time
    print("win_ratio:{:.2f}% mean_step:{:.2f}".format(wins/repeat_time * 100, step_mean))


if __name__ == '__main__':
    # main()
    # play_repeat(50)
    player_0 = MiniMaxTree(height=2, explore_num=5, policy_fun=policy_value_fn_1)
    player_1 = Greedy_Strategy()
    g = Game(board=Board())
    winner, data = g.get_self_play_data([player_0, player_0], True)
