# author='lwz'
# coding:utf-8
import random


class AI_Player(object):
    """AI player based on MCTS"""
    def __init__(self, strategy):
        self.strategy = strategy

    def set_player_ind(self, player_id):
        # 设置玩家ID
        self.player = player_id

    def reset_player(self):
        self.strategy.reset()

    def get_action(self, board):
        return self.strategy.get_action(board)


class Random_Strategy(object):
    def get_action(self, board):
        actions = board.get_availables()
        action = random.sample(actions, 1)[0]
        return action


class Greedy_Strategy(object):
    def action_evaluate(self, board, action):
        player_id = board.current_player_id
        (peg_id, target_loc) = action
        peg_loc = board.players_pegs[player_id, peg_id]
        (x, y) = board.get_location(peg_loc)
        (x_, y_) = board.get_location(target_loc)
        if player_id == 0:
            return (x_ + y_) - (x + y)
        else:
            return (x + y) - (x_ + y_)

    def get_action(self, board):
        actions = board.get_availables()
        scores = list()
        for action in actions:
            score = self.action_evaluate(board, action)
            scores.append(score)

        max_score = max(scores)
        actions_best = [actions[i] for i in range(len(scores)) if scores[i] >= max_score]
        action = random.sample(actions_best, 1)[0]
        return action
