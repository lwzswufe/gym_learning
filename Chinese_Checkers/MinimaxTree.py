# author='lwz'
# coding:utf-8
'''
Minimax tree
'''
import numpy as np
import copy
import random


def policy_value_fn_1(board, actions=None):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    if actions is None:
        actions = board.get_availables()

    scores = np.zeros(len(actions))
    for i, action in enumerate(actions):
        scores[i] = action_evaluate(board, action)

    scores -= np.min(scores)
    scores += 0.01
    scores = np.power(scores, 5)
    if np.sum(scores) == 0:
        print('sum(scores) is 0')

    action_probs = scores * (1 / max(np.sum(scores), 1))
    return action_probs


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


def policy_value_fn_0(board, actions=None):
    action_probs = np.ones(len(actions))
    return action_probs


class TreeNode(object):
    """
    节点
    A node in the minmax tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, height, policy_value_fn=None):
        self._parent = parent
        self._children = []  # a map from action to TreeNode
        self._height = height
        self._Q = - np.inf
        # self._P = prior_p  # 先验概率
        self._action = None

        if parent is not None:
            self._player_id = parent._player_id
            self.explore_num = parent.explore_num
            self.policy_value_fn = parent.policy_value_fn
        elif policy_value_fn is None:
            print('policy_value_fn is None')
        else:
            self.policy_value_fn = policy_value_fn

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
                return
        else:
            self._player_id = board.current_player_id

        if self._height <= 0:  # 节点高度小于等于0 终止拓展
            scores = np.array(board.get_score())
            self._Q = sum(scores[self._player_id] - scores)
            # board.graphic()
            return

        actions = board.get_availables()
        scores = np.zeros(len(actions))
        probs = self.policy_value_fn(board, actions)
        if isinstance(probs, tuple):
            # 解析 策略估值网络返回的数据
            probs, _ = probs
            probs_ = np.zeros(len(actions))
            for i, (action_, prob_) in enumerate(probs):
                probs_[i] = prob_
            probs = probs_
            probs /= np.sum(probs)

        if len(actions) > self.explore_num > 0:
            action_ids = np.random.choice(range(len(actions)), size=self.explore_num, p=probs, replace=False)
            scores = np.zeros(len(action_ids))
            actions_ = []
            for action_id in action_ids:
                actions_.append(actions[action_id])
            if len(actions_) == 0:
                print('actions_ is empty')
            else:
                actions = actions_

        for i, action in enumerate(actions):
            self._children.append(TreeNode(self, self._height - 1))
            board_copy = copy.deepcopy(board)
            self._children[i].expand(board_copy, action)
            scores[i] = self._children[i].get_value()

        if self._height % 2 == 0:
            self._Q = np.max(scores)
        else:
            self._Q = np.min(scores)

    def get_value(self):
        return self._Q


class MiniMaxTree(object):
    name = 'minimaxTree'

    def __init__(self, height=2, policy_fun=policy_value_fn_0, explore_num=5):
        self.height = height
        self.policy_fun = policy_fun
        self.explore_num = explore_num
        self.root = TreeNode(None, height, policy_fun)
        self.root.explore_num = explore_num

    def get_action(self, board, actions=None, probs=None):
        if actions is None:
            actions, probs = self.get_probs(board)

        action_id = np.random.choice(range(len(actions)), 1, p=probs)[0]
        action = actions[action_id]
        # board.graphic()
        # self.show_all_choose()
        self.reset()
        return action

    def get_probs(self, board):
        self.root.expand(board, None)
        actions = []
        scores = []
        for child in self.root._children:
            if child is None:
                pass
            elif True:
                scores.append(child._Q)
                actions.append(child._action)

        scores = np.array(scores, dtype=np.float)
        scores -= min(scores)
        scores += 0.01
        p = np.power(np.array(scores), 5)
        p = p / sum(p)
        return actions, p

    def show_all_choose(self):
        for child in self.root._children:
            print('action:{} Q:{}'.format(child._action, child._Q))

    def reset(self):
        self.root = TreeNode(None, self.height, self.policy_fun)
        self.root.explore_num = self.explore_num
