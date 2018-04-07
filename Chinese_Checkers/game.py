# author='lwz'
# coding:utf-8
from Chinese_Checkers.Board import Board
from Chinese_Checkers.MCTS import MCTS, MCTSPlayer
from Chinese_Checkers.Player import AI_Player, Random_Strategy, Greedy_Strategy
from Chinese_Checkers.MinimaxTree import MiniMaxTree, policy_value_fn_0, policy_value_fn_1
import time
import numpy as np


def self_play(board, players):
    flag = 0
    used_time = []
    board.init_board()
    for i, player in enumerate(players):
        player.player_id = i
        used_time.append(0)

    while True:
        time_st = time.time()
        player_id = flag % 2
        # board.players_pegs = np.array([
        #    [35, 29, 23, 34, 27, 33],
        #    [0, 6, 15, 1, 7, 2]], dtype=np.int)
        # board.update_state()
        action = players[player_id].get_action(board)
        board.do_move(action)
        board.graphic()
        end, winner = board.game_end()
        if end:
            print('game end the winner is: {}'.format(winner))
            break
        flag += 1
        time_ed = time.time()
        used_time[player_id] += time_ed - time_st

    print('usedtime: {:.4f}s  {:.4f}s'.format(used_time[0], used_time[1]))


def main():
    board = Board()

    # r_s = Random_Strategy()
    # player_0 = AI_Player(r_s)

    # player_0 = MCTSPlayer()
    player_0 = MiniMaxTree(height=4, policy_fun=policy_value_fn_1, threshold=0.2)
    # player_0 = MiniMaxTree(height=2)

    g_s = Greedy_Strategy()
    player_1 = AI_Player(g_s)
    # player_1 = MiniMaxTree(height=4)
    self_play(board, [player_0, player_1])


if __name__ == '__main__':
    main()