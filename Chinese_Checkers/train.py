# author='lwz'
# coding:utf-8
# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku
@author: Junxiao Song
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from Chinese_Checkers.Board import Board
from Chinese_Checkers.game import Game
from Chinese_Checkers.policy_value_tensorflow import PolicyValueNet  # Tensorflow
from Chinese_Checkers.MinimaxTree import MiniMaxTree, policy_value_fn_1
from Chinese_Checkers.MCTS import MCTSPlayer


class Train():
    def __init__(self, init_model, board):
        # params of the board and the game
        self.board_width = 6
        self.board_height = 6
        self.board_thick = 5
        self.n_in_row = 4
        self.board = board
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   self.board_thick,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   self.board_thick)
        self.ai_player = MiniMaxTree(height=2, explore_num=3, policy_fun=self.policy_value_net.policy_value_fn)

    def get_equi_data(self, play_data):
        """
        获取镜像数据
        augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.get_self_play_data([self.ai_player, self.ai_player], is_shown=False)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            # play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        start_probs_batch = [data[1][0].flatten() for data in mini_batch]
        end_probs_batch = [data[1][1].flatten() for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        start_probs, end_probs, old_v = self.policy_value_net.policy_value(state_batch)
        old_probs = np.hstack((start_probs, end_probs))

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    start_probs_batch,
                    end_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            start_probs, end_probs, new_v = self.policy_value_net.policy_value(state_batch)
            new_probs = np.hstack((start_probs, end_probs))
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=20):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_player = MiniMaxTree(policy_fun=self.policy_value_net.policy_value_fn,
                                          height=1,
                                          explore_num=10)
        # current_player = MCTSPlayer(c_puct=5, n_playout=10, policy_fun=self.policy_value_net.policy_value_fn)
        current_player.name = 'DQN player'
        pure_player = MiniMaxTree(policy_fun=self.policy_value_net.policy_value_fn,
                                         height=4,
                                         explore_num=5)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.self_play([current_player, pure_player],
                                          is_shown=False)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[0]) / n_games
        print("num_playouts:{}, win: {}, lose: {}".format(
                self.pure_mcts_playout_num,
                win_cnt[0], win_cnt[1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        # if (self.best_win_ratio == 1.0 and
                        #     self.pure_mcts_playout_num < 5000):
                        #     self.pure_mcts_playout_num += 1000
                        #     self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = Train(init_model=None, board=Board())
    training_pipeline.run()
