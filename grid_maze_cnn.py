# author='lwz'
# coding:utf-8

import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque  # 双端队列


# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch


class DQN():
    # DQN Agent
    def __init__(self, env, conv_1=[16, 5, 1], conv_2=[32, 5, 1], pool_1=[2, 2], pool_2=[3, 3]):
        # init experience replay
        self.replay_buffer = deque()                     # 缓存经验
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON

        self.state_dim = env.observation_space.shape[0]  # 状态数量
        self.action_dim = env.action_space.n             # 行动数量

        self.conv_1 = conv_1
        self.conv_2 = conv_2
        self.pool_1 = pool_1
        self.pool_2 = pool_2
        self.state_n = 10
        self.width = 0
        self.height = 0
        self.node_num_2 = (self.width / pool_1[1] / pool_2[1]) * \
                         (self.height / pool_1[0] / pool_2[0])

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def create_Q_network(self):
        tf_x = tf.placeholder(tf.float32, [None, self.height * self.width])
        image = tf.reshape(tf_x, [-1, self.height, self.width, 1])  # (batch, height, width, channel)
        #  -1 can also be used to infer推断 the shape
        tf_y = tf.placeholder(tf.int32, [None, self.state_n])  # input y
        # tf.nn.conv2d，一般在下载预训练好的模型时使用。

        conv1 = tf.layers.conv2d(inputs=image, filters=self.conv_1[0], kernel_size=self.conv_1[1],
                                 strides=self.conv_1[2], padding='same', activation=tf.nn.softmax)
        # shape (28, 28, 1)  第一组卷积层
        # inputs指需要做卷积的输入图像，它要求是一个Tensor
        # filters卷积核的数量
        # kernel_size: convolution window 卷积窗口 5*5
        # strides卷积时在图像每一维的步长，这是一个一维的向量，长度1
        # padding只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
        # 当其为‘SAME’时，表示卷积核可以停留在图像边缘
        # -> (28, 28, 16)
        # activation 正则化项

        pool1 = tf.layers.max_pooling2d(conv1, pool_size=self.pool_1[0], strides=self.pool_1[1])
        # 第一组池化层
        # the size of the pooling window 池化层大小2*2
        # 卷积时在图像每一维的步长，这是一个一维的向量，长度2
        # -> (14, 14, 16)

        conv2 = tf.layers.conv2d(pool1, self.conv_2[0], self.conv_2[1], self.conv_2[2]
                                 , 'same', activation=tf.nn.relu)
        # -> (14, 14, 32)
        pool2 = tf.layers.max_pooling2d(conv2, self.pool_2[0], self.pool_2[1])
        # -> (7, 7, 32)
        flat = tf.reshape(pool2, [-1, self.node_num_2 * self.conv_2[0]])  # -> (7*7*32, )
        cnn_output = tf.layers.dense(flat, self.state_n)  # output layer

        # tf.metrics.accuracy计算精度,返回accuracy和update_operation

        # network weights
        W1 = self.weight_variable([self.state_dim, self.state_n])
        b1 = self.bias_variable([self.state_n])
        W2 = self.weight_variable([self.state_n, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        # self.state_input = tf.placeholder("float", [None, self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(cnn_output, W1) + b1)
        # h = relu(W1 * state + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer, W2) + b2
        # q = W2 * h + b2

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        # tf.reduce_sum(reduction_indices=1)) 按行累计求和
        # tf.multiply  点乘
        # Q_action  q估计值
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        # self.y_input  q实际值
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        # 缓存经验
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        # 删除过多的经验
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        # 训练
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        # 从记忆中取样
        # data = (0state, 1action, 2reward, 3next_state, 4done)
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)

        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []   # q-learning的Q值
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:  # 终止状态
                y_batch.append(reward_batch[i])
            else:     # 中间状态
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
            })

    def egreedy_action(self, state):  # e 贪婪策略
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
            })[0]
        if random.random() <= self.epsilon:  # 随机选择
            return random.randint(0, self.action_dim - 1)
        else:                                # 贪婪策略
            return np.argmax(Q_value)

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000
        # epsilon 递减

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
            })[0])

    def weight_variable(self, shape):
        # 生成的值会遵循一个指定了平均值和标准差的正态分布，只保留
        # 两个标准差以内的值，超出的值会被弃掉重新生成。
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

# ---------------------------------------------------------
# ---------------------------------------------------------
# Hyper Parameters

ENV_NAME = 'GridMaze-v0'
EPISODE = 10000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = DQN(env)

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        for step in range(STEP):
            action = agent.egreedy_action(state)
            # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            # Define reward for agent
            if done:  # 终止状态
                reward = -1
            else:
                reward = 0.1

            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
            if ave_reward == 200:
                break


if __name__ == '__main__':
    main()