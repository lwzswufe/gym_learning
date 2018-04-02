# author='lwz'
# coding:utf-8
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)
"""
from MCTS import MCTSPlayer as Player
from Board_maze import Game, Board

if __name__ == '__main__':
    board = Board(barrier_ratio=0.4)
    player = Player()
    game = Game(board=board, player=player)
    game.start_play(is_shown=True)
