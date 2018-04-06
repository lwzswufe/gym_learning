# author='lwz'
# coding:utf-8
'''
Minimax tree
'''
import numpy as np
import copy
import random


class TreeNode(object):
    """
    节点
    A node in the minmax tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p, height):
        self._parent = parent
        self._children = []  # a map from action to TreeNode
        self._height = height
        self._Q = - np.inf
        self._P = prior_p  # 先验概率
        self._action = None
        if parent is not None:
            self._player_id = parent._player_id

    def expand(self, board, action):
        """
        扩展节点
        Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        if action is not None:
            self._action = action
            board.do_move(action)
            end, winner = board.game_end()
            if end:
                if winner == self._player_id:
                    self._Q = 100
                else:
                    scores = np.array(board.get_score())
                    self._Q = sum(scores[self._player_id] - scores)
        else:
            self._player_id = board.current_player_id

        if self._height <= 0:  # 节点高度小于等于0 终止拓展
            scores = np.array(board.get_score())
            self._Q = sum(scores[self._player_id] - scores)
            # board.graphic()
            return

        actions = board.get_availables()
        scores = np.zeros(len(actions))
        for i, action in enumerate(actions):
            board_copy = copy.deepcopy(board)
            self._children.append(TreeNode(self, 0, self._height - 1))
            self._children[i].expand(board_copy, action)
            scores[i] = self._children[i].get_value()

        if self._height % 2 == 0:
            self._Q = np.max(scores)
        else:
            self._Q = np.min(scores)

    def get_value(self):
        return self._Q


class MiniMaxTree(object):
    def __init__(self, height):
        self.height = height
        self.root = TreeNode(None, 0, height)

    def get_action(self, board):
        self.root.expand(board, None)
        actions = []
        for child in self.root._children:
            if child._Q == self.root._Q:
                actions.append(child._action)

        action = random.sample(actions, 1)[0]
        # board.graphic()
        # self.show_all_choose()
        self.reset()
        return action

    def show_all_choose(self):
        for child in self.root._children:
            print('action:{} Q:{}'.format(child._action, child._Q))

    def reset(self):
        self.root = TreeNode(None, 0, self.height)
