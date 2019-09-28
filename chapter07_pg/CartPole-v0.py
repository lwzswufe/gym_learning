# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import tensorflow.compat.v2 as tf
from tensorflow import keras


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

    def learn(self, observation, action, reward, done):
        '''
        学习函数
        observation  状态观测值 位置 速度
        action      动作
        reward      奖励
        done        是否完成
        next_action 下一动作
        '''
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


def play_montecarlo(env, agent, render=False, train=False):
    '''
    智能体环境交互逻辑
    env       环境
    agent     智能体
    train     是否训练
    render    是否显示
    返回期望收益
    '''
    observation = env.reset()
    episode_reward = 0.
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, done)
        if done:
            break
        observation = next_observation
    return episode_reward


class RandomAgent:
    '''
    随机策略
    '''
    def __init__(self, env):
        self.action_n = env.action_space.n

    def decide(self, observation):
        action = np.random.choice(self.action_n)
        return action


class OffPolicyVPGAgent(VPGAgent):
    '''
    回合更新策略异策策略梯度算法
    '''
    def __init__(self, env, policy_kwargs, baseline_kwargs=None, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma

        self.trajectory = []
        self.train_times = 0

        def dot(y_true, y_pred):
            return -tf.reduce_sum(y_true * y_pred, axis=-1)
        # 策略网络
        self.policy_net = self.build_network(output_size=self.action_n, output_activation=tf.nn.softmax, loss=dot, **policy_kwargs)
        # 基线网络
        if baseline_kwargs:
            self.baseline_net = self.build_network(output_size=1, **baseline_kwargs)

    def learn(self, observation, action, behavior, reward, done):
        '''
        学习函数
        observation  状态观测值 位置 速度
        action      动作
        reward      奖励
        behavior    其他策略的行为
        done        是否完成
        next_action 下一动作
        '''
        # 储存经验
        self.trajectory.append((observation, action, behavior, reward))
        # 若本次游戏完成
        if done:
            df = pd.DataFrame(self.trajectory, columns=['observation', 'action', 'behavior', 'reward'])
            df['discount'] = self.gamma ** df.index.to_series()
            df['discounted_reward'] = df['discount'] * df['reward']
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()
            df['psi'] = df['discounted_return']
            # 基线学习
            x = np.stack(df['observation'])
            if hasattr(self, 'baseline_net'):
                df['baseline'] = self.baseline_net.predict(x)
                df['psi'] -= df['baseline'] * df['discount']
                df['return'] = df['discounted_return'] / df['discount']
                y = df['return'].values[:, np.newaxis]
                self.baseline_net.fit(x, y, verbose=0)
            # 估计行动值
            y = np.eye(self.action_n)[df['action']] * (df['psi'] / df['behavior']).values[:, np.newaxis]
            # 策略学习
            self.policy_net.fit(x, y, verbose=0)
            # 为下一回合初始化经验列表
            self.train_times += 1
            self.trajectory = []


def train(env, agent, play_fun):
    # 训练
    episodes = 500
    episode_rewards = []
    for episode in range(episodes):
        episode_reward = play_fun(env, agent, train=True)
        episode_rewards.append(episode_reward)
    plt.plot(episode_rewards)
    plt.show()

    # 测试
    episode_rewards = [play_fun(env, agent, train=False) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))


def train_off_strategy(env, learn_agent, behavior_agent, play_fun):
    # 训练
    episodes = 500
    episode_rewards = []
    for episode in range(episodes):
        observation = env.reset()
        episode_reward = 0.
        while True:
            # 行为策略决策
            action = behavior_agent.decide(observation)
            # 重要性采样
            behavior = 1 / behavior_agent.action_n
            next_observation, reward, done, _ = env.step(action)
            episode_reward += reward
            # 策略学习
            learn_agent.learn(observation, action, behavior, reward, done)
            if done:
                break
            observation = next_observation
        # 跟踪监控
        episode_reward = play_montecarlo(env, learn_agent, train=False)
        episode_rewards.append(episode_reward)

    plt.plot(episode_rewards)
    plt.show()
    # 测试
    episode_rewards = [play_montecarlo(env, learn_agent, train=False) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))


def main():
    np.random.seed(0)
    tf.random.set_seed(0)
    env = gym.make('CartPole-v0')
    env.seed(0)

    print(">>>>>>>>>>>>>>>>>>>>>>不带基线的简单策略梯度算法<<<<<<<<<<<<<<<<<<<<<<")
    policy_kwargs = {'hidden_sizes': [10, ], 'activation': tf.nn.relu, 'learning_rate': 0.004}
    single_agent = VPGAgent(env, policy_kwargs=policy_kwargs)
    train(env, single_agent, play_montecarlo)

    print(">>>>>>>>>>>>>>>>>>>>>>带基线的简单策略梯度算法<<<<<<<<<<<<<<<<<<<<<<")
    policy_kwargs = {'hidden_sizes': [10, ], 'activation': tf.nn.relu, 'learning_rate': 0.001}
    baseline_kwargs = {'hidden_sizes': [10, ], 'activation': tf.nn.relu, 'learning_rate': 0.001}
    base_agent = VPGAgent(env, policy_kwargs=policy_kwargs, baseline_kwargs=baseline_kwargs)
    train(env, base_agent, play_montecarlo)

    print(">>>>>>>>>>>>>>>>>>>>>>不带基线的异策略梯度算法<<<<<<<<<<<<<<<<<<<<<<")
    policy_kwargs = {'hidden_sizes': [10, ], 'activation': tf.nn.relu, 'learning_rate': 0.02}
    off_agent = OffPolicyVPGAgent(env, policy_kwargs=policy_kwargs)
    random_agent = RandomAgent(env)
    train_off_strategy(env, off_agent, random_agent, play_montecarlo)

    print(">>>>>>>>>>>>>>>>>>>>>>带基线的异策略梯度算法<<<<<<<<<<<<<<<<<<<<<<")
    policy_kwargs = {'hidden_sizes': [10, ], 'activation': tf.nn.relu, 'learning_rate': 0.02}
    baseline_kwargs = {'hidden_sizes': [10, ], 'activation': tf.nn.relu, 'learning_rate': 0.03}
    base_off_agent = OffPolicyVPGAgent(env, policy_kwargs=policy_kwargs, baseline_kwargs=baseline_kwargs)
    train_off_strategy(env, base_off_agent, random_agent, play_montecarlo)

    print(">>>>>>>>>>>>>>>>>>>>>>使用训练好的智能体 带基线的异策略梯度算法<<<<<<<<<<<<<<<<<<<<<<")
    policy_kwargs = {'hidden_sizes': [10, ], 'activation': tf.nn.relu, 'learning_rate': 0.001}
    baseline_kwargs = {'hidden_sizes': [10, ], 'activation': tf.nn.relu, 'learning_rate': 0.001}
    # base_off_agent = OffPolicyVPGAgent(env, policy_kwargs=policy_kwargs, baseline_kwargs=baseline_kwargs)
    train_off_strategy(env, base_off_agent, base_agent, play_montecarlo)
    print("behavior_agent train_times:{} learn_agent train_times: {}".format(base_agent.train_times, base_off_agent.train_times))


if __name__ == "__main__":
    main()
