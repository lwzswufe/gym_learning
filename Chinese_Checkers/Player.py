# author='lwz'
# coding:utf-8
import random
import numpy as np


class AI_Player(object):
    name = 'default'
    player_id = None


class Random_Strategy(AI_Player):
    name = 'random'

    def get_action(self, board):
        actions = board.get_availables()
        action = random.sample(actions, 1)[0]
        return action


class Greedy_Strategy(AI_Player):
    name = 'greed'

    @staticmethod
    def action_evaluate(board, action):
        player_id = board.current_player_id
        (peg_id, target_loc) = action
        peg_loc = board.players_pegs[player_id, peg_id]
        (x, y) = board.get_location(peg_loc)
        (x_, y_) = board.get_location(target_loc)
        if player_id == 0:
            return (x_ + y_) - (x + y)
        else:
            return (x + y) - (x_ + y_)

    def get_probs(self, board):
        actions = board.get_availables()
        scores = np.zeros(len(actions))
        for i, action in enumerate(actions):
            score = self.action_evaluate(board, action)
            scores[i] = score

        scores -= np.min(scores)
        scores += 0.01
        probs = scores / sum(scores)
        return actions, probs

    def get_action(self, board, actions=None, probs=None):
        if actions is None:
            actions, probs = self.get_probs(board)

        max_prob = max(probs)
        actions_best = [i for i in range(len(probs)) if probs[i] >= max_prob]
        action_id = random.sample(actions_best, 1)[0]
        action = actions[action_id]
        return action

    def reset(self):
        pass


def get_prob(player, board=None):
    print(player.name)
    return [], []


if __name__ == "__main__":
    player = AI_Player()
    player.get_prob = get_prob
    print(player.get_prob())
