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
TRAIN_TIMES = 10000
BATCH_SIZE = 256  # size of minibatch


class DQN():
    # DQN Agent
    def __init__(self, env, conv_1=[16, 5, 1], conv_2=[32, 5, 1], pool_1=[2, 2], pool_2=[2, 2]):
        # init experience replay
        self.replay_buffer = deque()                     # 缓存经验
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_n = 10
        self.state_dim = self.state_n                    # 状态数量
        self.action_dim = len(env.env.actions)            # 行动数量

        self.conv_1 = conv_1
        self.conv_2 = conv_2
        self.pool_1 = pool_1
        self.pool_2 = pool_2

        self.width = env.env.width_cell
        self.height = env.env.height_cell
        self.node_num_2 = int(self.width / pool_1[1] / pool_2[1]) * \
                         int(self.height / pool_1[0] / pool_2[0])

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        self.state_input = tf.placeholder(tf.float32, [None, self.height * self.width])
        image = tf.reshape(self.state_input, [-1, self.height, self.width, 1])  # (batch, height, width, channel)
        #  -1 can also be used to infer推断 the shape

        # tf.nn.conv2d，一般在下载预训练好的模型时使用。

        conv1 = tf.layers.conv2d(inputs=image, filters=self.conv_1[0], kernel_size=self.conv_1[1],
                                 strides=self.conv_1[2], padding='same', activation=tf.nn.sigmoid)
        # shape (28, 28, 1)  第一组卷积层
        # inputs指需要做卷积的输入图像，它要求是一个Tensor
        # filters卷积核的数量
        # kernel_size: convolution window 卷积窗口 5*5
        # strides卷积时在图像每一维的步长，这是一个一维的向量，长度1
        # padding只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
        # 当其为‘SAME’时，表示卷积核可以停留在图像边缘
        # -> (28, 28, 16)
        # activation 正则化项

        pool1 = tf.layers.average_pooling2d(conv1, pool_size=self.pool_1[0], strides=self.pool_1[1])
        # 第一组池化层
        # the size of the pooling window 池化层大小2*2
        # 卷积时在图像每一维的步长，这是一个一维的向量，长度2
        # -> (14, 14, 16)

        conv2 = tf.layers.conv2d(pool1, self.conv_2[0], self.conv_2[1], self.conv_2[2]
                                 , 'same', activation=tf.nn.sigmoid)
        # -> (14, 14, 32)
        pool2 = tf.layers.max_pooling2d(conv2, self.pool_2[0], self.pool_2[1])
        # -> (7, 7, 32)
        # avgpool, maxpool

        flat = tf.reshape(pool2, [-1, self.node_num_2 * self.conv_2[0]])  # -> (7*7*32, )
        W0 = self.weight_variable([self.node_num_2 * self.conv_2[0], self.state_n])
        b0 = self.bias_variable([self.state_n])
        self.cnn_output = tf.nn.softmax(tf.matmul(flat, W0) + b0)

        # tf.metrics.accuracy计算精度,返回accuracy和update_operation

        # network weights
        W1 = self.weight_variable([self.state_dim, self.state_n])
        b1 = self.bias_variable([self.state_n])
        W2 = self.weight_variable([self.state_n, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        # self.state_input = tf.placeholder("float", [None, self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.cnn_output, W1) + b1)
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
        # cnn_class = self.cnn_output.eval(feed_dict={self.state_input:[state]})
        Q_value, cnn_class = self.session.run([self.Q_value, self.cnn_output], feed_dict={
            self.state_input: [state]})

        if random.random() <= self.epsilon:  # 随机选择
            return random.randint(0, self.action_dim - 1), cnn_class
        else:                                # 贪婪策略
            return np.argmax(Q_value), cnn_class

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / TRAIN_TIMES
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
EPISODE = 10000  # Episode limitation  总训练次数
STEP = 40  # Step limitation in an episode 最大步长
TEST = 20  # The number of experiment test every 100 episode 测试次数
state_mat = np.zeros([100, 10])


def get_maze(env, state):
    maze = env.env.maze
    maze[state] = 0.1
    return maze


def print_array(array, precision=4):
    context = ''
    for a in array:
        context += '|{:.4f} '.format(a)
    print(context)


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = DQN(env)

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()

        # Train
        for step in range(STEP):
            maze_now = get_maze(env, state)
            action, cnn_class = agent.egreedy_action(maze_now)
            state_mat[episode % 100, :] = cnn_class
            # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            maze_next = get_maze(env, next_state)
            # Define reward for agent
            if done:  # 终止状态
                reward = 1
            elif step == STEP - 1:
                reward = 0.08 - env.env.distance() / 100
            else:
                reward = -0.03

            agent.perceive(maze_now, action, reward, maze_next, done)
            # 存储经验--训练
            state = next_state

            if done:
                break

        # Test every 100 episodes
        if (episode + 1) % 10 == 0:
            print_array(np.mean(state_mat, axis=0))
            print_array(np.std(state_mat, axis=0))
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    # env.render()
                    maze_now = get_maze(env, state)
                    action = agent.action(maze_now)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print('episode: {} Reward: {:.4f}'.format(episode + 1, ave_reward))
            if ave_reward == 200:
                break


if __name__ == '__main__':
    main()