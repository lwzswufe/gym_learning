# author='lwz'
# coding:utf-8
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)
"""

import numpy as np
import copy
from operator import itemgetter
import random


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


def rollout_policy_fn(board):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    actions = board.get_availables()
    scores = np.zeros(len(actions))
    for i, action in enumerate(actions):
        scores[i] = action_evaluate(board, action)

    scores -= np.min(scores)
    scores += 0.01
    scores = np.power(scores, 2)
    if np.sum(scores) == 0:
        print('err')

    action_probs = scores * (1 / max(np.sum(scores), 1))
    return zip(actions, action_probs)


def policy_value_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    actions = board.get_availables()
    action_probs = np.ones(len(actions))/len(actions)
    return zip(actions, action_probs), 0


def get_max_action(actions):
    samples = []
    max_value = - np.inf
    for action in actions:
        if action[1] > max_value:
            samples = []
            samples.append(action)
        elif action[1] == max_value:
            samples.append(action)
    if len(samples) == 0:
        print('')
    return random.sample(samples, 1)[0]


class TreeNode(object):
    """
    节点
    A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """
        扩展节点
        Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        选择行为
        Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        更新Q值 加权平均
        Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        # N * Q_ = (N - 1) * Q + Q_new

    def update_recursive(self, leaf_value):
        """
        Like a call to update(), but applied recursively for all ancestors.
        在Nn的模拟结束之后，它的父节点N以及从根节点到N的路径上的所有节点
        都会根据本次模拟的结果来添加自己的累计评分。
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
            一个正常数 控制探索行为收敛到最优策略的速度 这个值越高，收敛速度就越快
        n_playout： 模拟运行打次数
        """
        self._root = TreeNode(None, 1.0)  # 当前节点  当前位置
        # self._policy = policy_value_fn
        self._policy = rollout_policy_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.limit = 40

    def _playout(self, board):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():

                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)  # 1选择
            board.do_move(action)
            # board.graphic()

        # actions = board.get_availables()
        action_probs = self._policy(board)
        # Check for end of game
        end, winner = board.game_end()
        if not end:
            node.expand(action_probs)                 # 2拓展
        else:
            pass
            # print('end')
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(board)    # 3模拟
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)            # 4回溯

    def _evaluate_rollout(self, board):
        """
        # 模拟我们从Nn开始，让游戏随机进行，直到得到一个游戏结局，
        这个结局将作为Nn的初始评分。一般使用胜利/失败来作为评分，只有1或者0。
        Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        player = board.current_player_id
        winner = -1
        for i in range(self.limit):
            end, winner = board.game_end()
            if end:
                break

            # action_probs = rollout_policy_fn(board)
            action_probs = self._policy(board)
            max_action = get_max_action(action_probs)[0]
            board.do_move(max_action)
            # board.graphic()
        else:
            pass
            # If no break from the loop, issue a warning.
            # print("WARNING: rollout reached move limit")

        if winner < 0:  # tie 平局
            score = np.array(board.get_score())
            return np.mean(score[player] - score) / 30
        elif winner == player:
            return 1
        else:
            return -1

    def get_action(self, board):
        """
        Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action [move, TreeNode]
        """
        for n in range(self._n_playout):
            board_copy = copy.deepcopy(board)
            self._playout(board_copy)

        if len(board.get_availables()) != len(self._root._children):
            print('err')

        action, Q_ = self.get_prob()
        return action

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children.keys():
            self._root = self._root._children[last_move]
            self._root._parent = None
            actions = self._root._children.keys()
            if len(actions) < 0:
                print(self._root._children.keys())
            pass
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

    def get_prob(self):
        scores = []
        actions = []
        score_max = -np.inf
        for action in self._root._children.keys():
            node = self._root._children[action]
            Q = node._Q
            if node._n_visits == 0:
                continue
            if Q > score_max:
                scores = [Q]
                actions = [action]
            elif Q == score_max:
                scores.append(Q)
                actions.append(action)

        # print(Q)
        action_id = random.randint(0, len(scores) - 1)
        action, Q_ = actions[action_id], scores[action_id]
        return action, Q_


class MCTSPlayer(object):
    """AI player based on MCTS"""
    name = 'MCTS'

    def __init__(self, c_puct=5, n_playout=1000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, player_id):
        # 设置玩家ID
        self.player = player_id

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        self.mcts.update_with_move(board.last_move)
        action = self.mcts.get_action(board)
        self.mcts.update_with_move(action)
        return action

    def __str__(self):
        return "MCTS {}".format(self.player)


