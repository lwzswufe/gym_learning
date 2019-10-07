# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import tensorflow.compat.v2 as tf
from tensorflow import keras


class DQNReplayer:
    '''
    深度 Q 网络求解最优策略
    经验回放
    '''
    def __init__(self, capacity):
        '''
        capacity 经验储存条数
        '''
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation', 'action', 'reward', 'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        '''
        储存一条经验
        args = (observation, action, reward, next_observation, done)
        '''
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        '''
        随机选取size条经验
        返回多个数组
        observation_arr, action_arr, reward_arr, next_observation_arr, done_arr
        '''
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)


class DQNAgent:
    '''
    深度Q学习智能体
    '''
    def __init__(self, env, net_kwargs={}, gamma=0.99, epsilon=0.001,
                 replayer_capacity=10000, batch_size=64):
        '''
        env                 环境
        net_kwargs          神经网络参数
        gamma               衰减系数
        epsilon             柔性策略概率
        replayer_capacity   经验储存条数
        batch_size          批处理规模
        '''
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)  # 经验回放
        self.evaluate_net = self.build_network(input_size=observation_dim, output_size=self.action_n, **net_kwargs)  # 评估网络
        self.target_net = self.build_network(input_size=observation_dim, output_size=self.action_n, **net_kwargs)  # 目标网络
        self.target_net.set_weights(self.evaluate_net.get_weights())

    def build_network(self, input_size, hidden_sizes, output_size,
                      activation=tf.nn.relu, output_activation=None,
                      learning_rate=0.01):
        '''
        构建keras网络
        '''
        model = keras.Sequential()
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = dict(input_shape=(input_size,)) if not layer else {}
            model.add(keras.layers.Dense(units=hidden_size, activation=activation, **kwargs))

        model.add(keras.layers.Dense(units=output_size, activation=output_activation))  # 输出层
        optimizer = tf.optimizers.Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def learn(self, observation, action, reward, next_observation, done):
        '''
        训练
        '''
        # 存储经验
        self.replayer.store(observation, action, reward, next_observation, done)
        # 经验回放
        observations, actions, rewards, next_observations, dones = self.replayer.sample(self.batch_size)
        # 目标网络计算学习的目标 us
        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * (1. - dones) * next_max_qs
        # 评估网路 向目标us学习
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if done:  # 更新目标网络
            self.target_net.set_weights(self.evaluate_net.get_weights())

    def decide(self, observation):
        '''
        epsilon贪心策略
        '''
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        qs = self.evaluate_net.predict(observation[np.newaxis])
        return np.argmax(qs)


def play_qlearning(env, agent, train=False, render=False):
    '''
    Q学习环境交互
    env     环境
    agent   智能体
    train   是否训练
    render  是否显示
    '''
    episode_reward = 0
    observation = env.reset()
    done = False
    while not done:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, next_observation, done)
        observation = next_observation
    return episode_reward


class DoubleDQNAgent(DQNAgent):
    '''
    双重深度 Q 网络
    '''
    def learn(self, observation, action, reward, next_observation, done):
        # 存储经验
        self.replayer.store(observation, action, reward, next_observation, done)
        # 经验回放
        observations, actions, rewards, next_observations, dones = self.replayer.sample(self.batch_size)
        # 评估网路 确定动作
        next_eval_qs = self.evaluate_net.predict(next_observations)
        next_actions = next_eval_qs.argmax(axis=-1)
        # 目标网络计算学习的目标 us 回报
        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs[np.arange(next_qs.shape[0]), next_actions]
        us = rewards + self.gamma * next_max_qs * (1. - dones)
        # 评估网路 向目标us学习
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if done:
            self.target_net.set_weights(self.evaluate_net.get_weights())


class VPGAgent:
    '''
    回合更新策略梯度算法寻找最优策略
    '''
    def __init__(self, env, policy_kwargs, baseline_kwargs=None, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.trajectory = []
        self.train_times = 0
        self.policy_net = self.build_network(output_size=self.action_n,
                                             output_activation=tf.nn.softmax,
                                             loss=tf.losses.categorical_crossentropy,
                                             **policy_kwargs)
        # 若是基线策略
        if baseline_kwargs:
            self.baseline_net = self.build_network(output_size=1, **baseline_kwargs)

    def build_network(self, hidden_sizes, output_size,
                      activation=tf.nn.relu, output_activation=None,
                      loss=tf.losses.mse, learning_rate=0.01):
        '''
        创建网络
        '''
        model = keras.Sequential()
        for hidden_size in hidden_sizes:
            model.add(keras.layers.Dense(units=hidden_size, activation=activation))
        model.add(keras.layers.Dense(units=output_size, activation=output_activation))
        optimizer = tf.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def decide(self, observation):
        '''
        决策函数
        @param: observation 观测值
        return  action      动作
        '''
        probs = self.policy_net.predict(observation[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, next_observation, done):
        '''
        学习函数
        observation  状态观测值 位置 速度
        action      动作
        reward      奖励
        next_observation 下一观测值 暂不使用
        done        是否完成
        '''
        _ = next_observation
        self.trajectory.append((observation, action, reward))
        # 如果程序结束 储存数据
        if done:
            df = pd.DataFrame(self.trajectory, columns=['observation', 'action', 'reward'])
            df['discount'] = self.gamma ** df.index.to_series()
            df['discounted_reward'] = df['discount'] * df['reward']
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()
            df['psi'] = df['discounted_return']

            x = np.stack(df['observation'])
            # 更新基线
            # np.newaxis的作用就是在这一位置增加一个维度 例如一维向量变二维矩阵
            # 基线的学习目标是 未来回报的贴现值
            if hasattr(self, 'baseline_net'):
                df['baseline'] = self.baseline_net.predict(x)
                df['psi'] -= (df['baseline'] * df['discount'])
                df['return'] = df['discounted_return'] / df['discount']
                y = df['return'].values[:, np.newaxis]
                self.baseline_net.fit(x, y, verbose=0)
            # 估计行动值
            y = np.eye(self.action_n)[df['action']] * df['psi'].values[:, np.newaxis]
            # 策略学习
            self.policy_net.fit(x, y, verbose=0)
            self.train_times += 1
            # 为下一回合初始化经验列表
            self.trajectory = []


def train(env, agent, play_fun):
    # 训练
    episodes = 500
    episode_rewards = []
    for episode in range(episodes):
        episode_reward = play_fun(env, agent, train=True)
        episode_rewards.append(episode_reward)
        print("{}/{} score:{:.2f}".format(episode, episodes, episode_reward))
    plt.plot(episode_rewards)
    plt.show()

    # 测试
    agent.epsilon = 0.  # 取消探索
    episode_rewards = [play_fun(env, agent, train=False) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))


def main():
    np.random.seed(0)
    tf.random.set_seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)
    print('观测空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print('位置范围 = {}'.format((env.unwrapped.min_position, env.unwrapped.max_position)))
    print('速度范围 = {}'.format((-env.unwrapped.max_speed, env.unwrapped.max_speed)))
    print('目标位置 = {}'.format(env.unwrapped.goal_position))

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>DQN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    net_kwargs = {'hidden_sizes': [16, ], 'learning_rate': 0.01}
    agent_dqn = DQNAgent(env, net_kwargs=net_kwargs)
    train(env, agent_dqn, play_qlearning)
    # play_qlearning(env, agent_dqn, render=True)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>双重深度 Q 网络<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    net_kwargs = {'hidden_sizes': [16, ], 'learning_rate': 0.004}
    agent_2_dqn = DoubleDQNAgent(env, net_kwargs=net_kwargs)
    train(env, agent_2_dqn, play_qlearning)

    # print(">>>>>>>>>>>>>>>>>>>>>>VPG 不带基线的简单策略梯度算法<<<<<<<<<<<<<<<<<<<<<<")
    # policy_kwargs = {'hidden_sizes': [10, ], 'activation': tf.nn.relu, 'learning_rate': 0.01}
    # agent_single = VPGAgent(env, policy_kwargs=policy_kwargs)
    # train(env, agent_single, play_qlearning)

    # print(">>>>>>>>>>>>>>>>>>>>>>带基线的简单策略梯度算法<<<<<<<<<<<<<<<<<<<<<<")
    # policy_kwargs = {'hidden_sizes': [10, ], 'activation': tf.nn.relu, 'learning_rate': 0.001}
    # baseline_kwargs = {'hidden_sizes': [10, ], 'activation': tf.nn.relu, 'learning_rate': 0.001}
    # base_agent = VPGAgent(env, policy_kwargs=policy_kwargs, baseline_kwargs=baseline_kwargs)
    # train(env, base_agent, play_qlearning)


if __name__ == "__main__":
    main()
