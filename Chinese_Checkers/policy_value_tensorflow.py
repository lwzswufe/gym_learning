# author='lwz'
# coding:utf-8
import numpy as np
import tensorflow as tf


class PolicyValueNet(object):
    def __init__(self, board_width, board_height, board_thick, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.board_thick = board_thick

        # Define the tensorflow neural network
        # 1. Input:
        self.input_states = tf.placeholder(
                tf.float32, shape=[None, board_thick, board_height, board_width])
        self.input_states_reshaped = tf.reshape(
                self.input_states, [-1, board_height, board_width, board_thick])
        # [batch, in_height, in_width, in_channels]
        # [图片数量, 图片高度, 图片宽度, 图像通道数]
        # 2. Common Networks Layers
        # 公共的3层全卷积网络
        self.conv1 = tf.layers.conv2d(inputs=self.input_states_reshaped,
                                      filters=32, kernel_size=[3, 3],
                                      padding="same", activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64,
                                      kernel_size=[3, 3], padding="same",
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128,
                                      kernel_size=[3, 3], padding="same",
                                      activation=tf.nn.relu)
        # 3-1 Action Networks+++++++++++++++++++++++++++++++++++++++++++++++++
        self.action_start_conv = tf.layers.conv2d(inputs=self.conv3, filters=4,
                                            kernel_size=[1, 1], padding="same",
                                            activation=tf.nn.relu)

        self.action_start_conv_flat = tf.reshape(
                self.action_start_conv, [-1, 4 * board_height * board_width])

        self.action_start_fc = tf.layers.dense(inputs=self.action_start_conv_flat,
                                         units=board_height * board_width,
                                         activation=tf.nn.log_softmax)

        self.action_end_conv = tf.layers.conv2d(inputs=self.conv3, filters=4,
                                                  kernel_size=[1, 1], padding="same",
                                                  activation=tf.nn.relu)
        self.action_end_conv_flat = tf.reshape(
            self.action_end_conv, [-1, 4 * board_height * board_width])

        self.action_end_fc = tf.layers.dense(inputs=self.action_end_conv_flat,
                                         units=board_height * board_width,
                                         activation=tf.nn.log_softmax)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 4 Evaluation Networks策略价值网络

        self.evaluation_conv = tf.layers.conv2d(inputs=self.conv3, filters=2,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                activation=tf.nn.relu)

        self.evaluation_conv_flat = tf.reshape(
                self.evaluation_conv, [-1, 2 * board_height * board_width])
        # 接一个64个神经元的全连接层
        self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                              units=64, activation=tf.nn.relu)
        # output the score of evaluation on current state
        # 最后使用tanh非线性函数直接输出 [-1,1] 之间的局面评分。
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                              units=1, activation=tf.nn.tanh)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Define the Loss function
        # 1. Label: the array containing if the game wins or not for each state
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 接一个64个神经元的全连接层，
        # 用来训练策略价值网络的是我们在self-play过程中收集的一系列的 (s,pi, z) 数据。
        # 根据上面的策略价值网络训练示意图，我们训练的目标是让
        #  策略价值网络输出的局面评分 v 能更准确的预测真实的对局结果 z
        self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                       self.evaluation_fc2)
        # 3-2. Policy Loss function
        # 我们训练的目标是让策略价值网络输出的action概率 p更加接近MCTS输出的概率 pi
        self.start_probs = tf.placeholder(
            tf.float32, shape=[None, board_height * board_width])
        self.end_probs = tf.placeholder(
                tf.float32, shape=[None, board_height * board_width])

        self.start_probs_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.start_probs, self.action_start_fc), 1)
                            ))
        self.end_probs_loss = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.multiply(self.end_probs, self.action_end_fc), 1)
                            ))
        # 3-3. L2 penalty (regularization)
        # 防止过拟合的正则项
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + self.start_probs_loss + self.end_probs_loss + l2_penalty

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # Make a session
        self.session = tf.Session()

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_start_fc) * self.action_start_fc, 1))) +\
                        tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_end_fc) * self.action_end_fc, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)

    def policy_value(self, state_batch):
        """
        输入状态 输出落子概率
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        log_start_probs, log_end_probs, value = self.session.run(
                [self.action_start_fc, self.action_start_fc, self.evaluation_fc2],
                feed_dict={self.input_states: state_batch}
                )
        start_probs = np.exp(log_start_probs)
        end_probs = np.exp(log_end_probs)
        return start_probs, end_probs, value

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        传送估值网络给 mcts_player
        """
        action_with_locs = board.availables_with_loc()
        # 列表 可下子点
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, self.board_thick, self.board_width, self.board_height))
        start_probs, end_probs, value = self.policy_value(current_state)

        probs = np.zeros(len(action_with_locs))
        actions = []
        for i, action in enumerate(action_with_locs):
            peg_id, end_loc_id, start_loc, end_loc = action
            probs[i] = min(start_probs[start_loc], end_probs[end_loc])
            actions.append(peg_id, end_loc_id)

        act_probs = zip(actions, probs)
        # act_probs  (point, prob) (点， 概率对)
        return act_probs, value

    def train_step(self, state_batch, probs, winner_batch, lr):
        """perform a training step"""
        start_probs, end_probs = probs
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
                [self.loss, self.entropy, self.optimizer],
                feed_dict={self.input_states: state_batch,
                           self.start_probs: start_probs,
                           self.end_probs: end_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr})
        return loss, entropy

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)