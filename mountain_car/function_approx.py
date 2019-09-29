# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mountain_car_env import MountainCarEnv


class TileCoder:
    '''
    砖瓦编码
    '''
    def __init__(self, layers, features):
        '''
        layers     要用到几层砖瓦编码
        features   砖瓦编码应该得到多少特征
        '''
        self.layers = layers
        self.features = features
        self.codebook = {}

    def get_feature(self, codeword):
        '''
        codeword 数据坐标(层数 坐标 坐标 动作)
        '''
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count >= self.features:  # 冲突处理
            return hash(codeword) % self.features
        else:
            self.codebook[codeword] = count
            return count

    def __call__(self, floats=(), ints=()):
        '''
        将观测值向量转化为 坐标
        floats 特征值 向量  即观测值映射从[下界,上界]到[0, 1]的映射 分位数
        ints   动作
        返回 features 不同层次的编码的位置1*feature_num的向量
        '''
        scaled_floats = tuple(f * self.layers * self.layers for f in floats)
        features = []
        for layer in range(self.layers):
            codeword = (layer,) + tuple(int((f + layer) / self.layers) for f in scaled_floats) + ints
            feature = self.get_feature(codeword)
            features.append(feature)
        return features


class SARSAAgent:
    def __init__(self, env, layers=8, features=1893, gamma=1., learning_rate=0.03, epsilon=0.001):
        '''
        学习函数
        env        环境
        layers     要用到几层砖瓦编码
        features   砖瓦编码应该得到多少特征 总特征数 8*8 + (8+1) * (8+1) * (8-1)
        gamma      收益衰减速率
        learning_rate 学习速率
        epsilon    执行探索策略概率
        '''
        self.action_n = env.action_space.n  # 动作数
        self.obs_low = env.observation_space.low
        self.obs_scale = env.observation_space.high - env.observation_space.low  # 观测空间范围
        self.encoder = TileCoder(layers, features)  # 砖瓦编码器
        self.w = np.zeros(features)  # 权重
        self.gamma = gamma  # 折扣
        self.learning_rate = learning_rate  # 学习率
        self.epsilon = epsilon  # 探索

    def encode(self, observation, action):
        '''
        将观测值编码为数据坐标值 index
        observation 状态观测值
        action      动作
        '''
        states = tuple((observation[:2] - self.obs_low) / self.obs_scale)
        actions = (action,)
        return self.encoder(states, actions)

    def get_q(self, observation, action):
        '''
        获取动作价值
        observation  状态观测值 (位置 速度 时间)
        action       动作 0, 1, 2
        '''
        features = self.encode(observation[:2], action)
        return self.w[features].sum()

    def decide(self, observation):
        '''
        决策函数
        observation  状态观测值 (位置 速度 时间)
        '''
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        else:
            qs = [self.get_q(observation, action) for action in range(self.action_n)]
            return np.argmax(qs)

    def learn(self, observation, action, reward, time_interval, next_observation, done, next_action):
        '''
        学习函数
        observation  状态观测值 位置 速度
        action      动作
        reward      奖励
        next_state  下一状态 观测值
        done        是否完成
        next_action 下一动作
        '''
        u = reward + (1. - done) * self.gamma * self.get_q(next_observation, next_action)
        td_error = u - self.get_q(observation, action)
        features = self.encode(observation, action)
        self.w[features] += (self.learning_rate * td_error * time_interval)


class SARSALambdaAgent(SARSAAgent):
    '''
    SARSA(λ)算法
    '''
    def __init__(self, env, layers=8, features=1893, gamma=1.,
                 learning_rate=0.03, epsilon=0.001, lambd=0.9):
        super().__init__(env=env, layers=layers, features=features,
                         gamma=gamma, learning_rate=learning_rate, epsilon=epsilon)
        self.lambd = lambd
        self.z = np.zeros(features)  # 初始化资格迹

    def learn(self, observation, action, reward, time_interval, next_observation, done, next_action):
        u = reward
        if not done:
            u += (self.gamma * self.get_q(next_observation, next_action))
            self.z *= (self.gamma * self.lambd)
            features = self.encode(observation, action)
            self.z[features] = 1.  # 替换迹
        td_error = u - self.get_q(observation, action)
        self.w += (self.learning_rate * td_error * self.z * time_interval)
        if done:
            self.z = np.zeros_like(self.z)  # 为下一回合初始化资格迹


def play_sarsa(env, agent, train=False, render=False, collect_ex=False):
    '''
    智能体环境交互逻辑
    env       环境
    agent     智能体
    train     是否训练
    render    是否显示
    返回期望收益
    '''
    episode_reward = 0
    trajectory = []
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, done, time_interval = env.step(action)
        episode_reward += reward
        next_action = agent.decide(next_observation)  # 终止状态时此步无意义
        if train:
            agent.learn(observation, action, reward, time_interval, next_observation, done, next_action)
        if collect_ex:  # 搜集经验
            trajectory.append((observation, action, reward, time_interval, next_observation, done, next_action))
        if done:
            break
        observation, action = next_observation, next_action
    return episode_reward, trajectory


def train(env, agent, play_fun):
    # 训练
    env.continuous_time = True
    episodes = 500
    episode_rewards = []
    for episode in range(episodes):
        episode_reward, _ = play_fun(env, agent, train=True)
        episode_rewards.append(episode_reward)
        # print("train {}/{}".format(episode + 1, episodes))
    plt.plot(episode_rewards)
    plt.show()

    # 测试
    env.continuous_time = False
    agent.epsilon = 0.0
    episodes = 100
    sum_rewards = 0.0
    for episode in range(episodes):
        sum_rewards += play_fun(env, agent, train=False)
        # print("train {}/{}".format(episode + 1, episodes))

    print('平均回合奖励 = {} / {} = {}'.format(sum_rewards, episodes, sum_rewards / episodes))


def train_off_line(env, learn_agent, behavior_agent, play_fun):
    # 训练
    env.continuous_time = True
    episodes = 500
    episode_rewards = []
    trajectory_list = []
    for episode in range(episodes):
        episode_reward, trajectory = play_fun(env, learn_agent, train=True)
        if episode >= 100:
            episode_rewards.append(episode_reward)
        trajectory_list += trajectory
        # print("train {}/{}".format(episode + 1, episodes))
    plt.plot(episode_rewards)
    plt.show()

    # 测试
    env.continuous_time = False
    agent.epsilon = 0.0
    episodes = 100
    sum_rewards = 0.0
    for episode in range(episodes):
        sum_rewards += play_fun(env, agent, train=False)
        # print("train {}/{}".format(episode + 1, episodes))

    print('平均回合奖励 = {} / {} = {}'.format(sum_rewards, episodes, sum_rewards / episodes))

def main():
    np.random.seed(0)
    env = MountainCarEnv()
    env.seed(0)

    sarsa_agent = SARSAAgent(env)
    print(">>>>>>>>>>>>>>>>>>>>SARSA 算法<<<<<<<<<<<<<<<<<<<<<")
    train(env, sarsa_agent, play_sarsa)

    lambda_agent = SARSALambdaAgent(env)
    print(">>>>>>>>>>>>>>>>>>>>SARSA(λ) 算法<<<<<<<<<<<<<<<<<<<<<")
    train(env, lambda_agent, play_sarsa)


if __name__ == "__main__":
    main()
