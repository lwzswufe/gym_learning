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


def self_play(board, players, is_shown=False):
    step = 0
    used_time = []
    board.init_board()
    for i, player in enumerate(players):
        player.player_id = i
        used_time.append(0)

    while True:
        time_st = time.time()
        player_id = step % len(players)
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

    g_s = Greedy_Strategy()
    player_1 = Greedy_Strategy
    # player_1 = MCTSPlayer(c_puct=30, n_playout=100)
    # player_1 = MiniMaxTree(height=4)
    self_play(board, [player_0, player_1], True)


def play_repeat(repeat_time):
    # player_0 = MiniMaxTree(height=2, policy_fun=policy_value_fn_1, explore_num=5)
    player_0 = MCTSPlayer(c_puct=5, n_playout=100)
    # player_0 = MiniMaxTree(height=2)

    player_1 = Greedy_Strategy()
    # player_1 = MiniMaxTree(height=4)
    wins = 0
    step_mean = 0
    for i in range(repeat_time):
        board = Board()
        if i % 2 == 0:
            winner, step = self_play(board, [player_0, player_1], False)
            wins += winner
        else:
            winner, step = self_play(board, [player_1, player_0], False)
            wins += (1 - winner)
        step_mean += step

    step_mean /= repeat_time
    print("win_ratio:{:.2f}% mean_step:{:.2f}".format(wins/repeat_time * 100, step_mean))


if __name__ == '__main__':
    # main()
    play_repeat(50)