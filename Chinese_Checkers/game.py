# author='lwz'
# coding:utf-8
from Chinese_Checkers.Board import Board
from Chinese_Checkers.MCTS import MCTS, MCTSPlayer
from Chinese_Checkers.Player import AI_Player, Random_Strategy, Greedy_Strategy
from Chinese_Checkers.MinimaxTree import MiniMaxTree


def self_play(board, players):
    flag = 0

    board.init_board()
    for i, player in enumerate(players):
        player.player_id = i

    while True:
        player_id = flag % 2
        action = players[player_id].get_action(board)
        board.do_move(action)
        board.graphic()
        end, winner = board.game_end()
        if end:
            print('game end the winner is: {}'.format(winner))
            break
        flag += 1


def main():
    board = Board()

    # r_s = Random_Strategy()
    # player_0 = AI_Player(r_s)

    # player_0 = MCTSPlayer()
    player_0 = MiniMaxTree(height=2)

    # g_s = Greedy_Strategy()
    # player_1 = AI_Player(g_s)
    player_1 = MiniMaxTree(height=4)
    self_play(board, [player_0, player_1])


if __name__ == '__main__':
    main()