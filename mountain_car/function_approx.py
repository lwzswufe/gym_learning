# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mountain_car_env import MountainCarEnv
from random import sample, seed


class GridCoder:
    '''
    网格编码
    '''
    def __init__(self, size_arr, obs_low, obs_scale):
        '''
        不同维度上的网格大小
        size_arr
        '''
        self.size_arr = np.array(size_arr)
        self.dim_num = len(self.size_arr)
        self.dim_w = np.ones(self.dim_num)
        self.obs_low = obs_low
        self.obs_scale = obs_scale
        w = 1
        for i, axis_size in enumerate(reversed(size_arr)):
            self.dim_w[self.dim_num - i - 1] = w
            w *= axis_size
        self.state_num = w

    def __call__(self, states=()):
        '''
        将观测值向量转化为 坐标
        states 特征值 向量  即观测值映射从[下界,上界]到[0, 1]的映射 分位数
        action 动作
        返回 Q值坐标 (position, action)
        '''
        position = np.floor((np.array(states[:2]) - self.obs_low) / self.obs_scale * self.size_arr) @ self.dim_w
        return int(position)


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


class Agent:
    '''
    智能体
    '''
    def __init__(self, env):
        self.action_n = env.action_space.n  # 动作数

    def decide(self, observation):
        '''
        决策函数
        observation  状态观测值 (位置 速度 时间)
        '''
        return np.random.randint(self.action_n)

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
        pass


class RandomAgent(Agent):
    '''
    随机决策智能体
    '''
    def decide(self, observation):
        '''
        决策函数
        observation  状态观测值 (位置 速度 时间)
        '''
        return np.random.randint(self.action_n)


class SARSAAgent(Agent):
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


class DiffAgent(SARSAAgent):
    '''
    异策略回合更新策略寻找最优策略
    '''
    def __init__(self, env, layers=8, features=1893, gamma=1., learning_rate=0.03, epsilon=0.001):
        SARSAAgent.__init__(self, env, layers, features, gamma, learning_rate, epsilon)
        self.c = np.zeros_like(self.w)
        self.layer_n = layers

    def learn(self, trajectory):
        '''
        学习函数
        observation  状态观测值 位置 速度
        action      动作
        reward      奖励
        next_state  下一状态 观测值
        done        是否完成
        next_action 下一动作
        '''
        # rho = np.ones(self.layer_n)
        for observation, action, reward, time_interval, next_observation, done, next_action in reversed(trajectory):
            u = reward + (1. - done) * self.gamma * self.get_q(next_observation, next_action)
            td_error = u - self.get_q(observation, action)
            features = self.encode(observation, action)
            self.w[features] += (self.learning_rate * td_error * time_interval)
            # features = self.encode(observation, action)
            # self.c[features] += np.ones(self.layer_n) / self.layers
            # self.w[features] += rho / self.c[features] * (reward - self.w[features].sum())
            # rho *= (self.w[features] / self.w[features])


class SARSAAgent_grid(Agent):
    '''
    网格太细会带来训练速度慢的问题 本质是q值表更新的不充分 不能从临近的网格获取数据
    '''
    def __init__(self, env, dim_size_arr=[50, 50], gamma=1., learning_rate=0.03, epsilon=0.001):
        '''
        学习函数
        env        环境
        dim_size_arr 各个维度上的网格数量
        gamma      收益衰减速率
        learning_rate 学习速率
        epsilon    执行探索策略概率
        '''
        self.action_n = env.action_space.n  # 动作数
        self.obs_low = env.observation_space.low
        self.obs_scale = env.observation_space.high - env.observation_space.low  # 观测空间范围
        self.encoder = GridCoder(dim_size_arr, self.obs_low, self.obs_scale)
        self.w = np.zeros((self.encoder.state_num, self.action_n))  # Q值
        self.c = np.zeros_like(self.w)
        self.w += np.random.rand(self.encoder.state_num, self.action_n) * 0.01
        self.gamma = gamma  # 折扣
        self.learning_rate = learning_rate  # 学习率
        self.epsilon = epsilon  # 探索

    def decide(self, observation):
        '''
        决策函数
        observation  状态观测值 (位置 速度 时间)
        '''
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        else:
            qs = self.w[self.encoder(observation), :]
            action = np.argmax(qs)
            return action

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
        u = reward + (1. - done) * self.gamma * self.w[self.encoder(next_observation), next_action]
        td_error = u - self.w[self.encoder(observation), action]
        self.w[self.encoder(observation), action] += (self.learning_rate * td_error * time_interval)


class VPGAgent(Agent):
    '''
    回合更新策略梯度算法寻找最优策略
    '''
    def __init__(self, env, layers=8, features=1893, gamma=1., learning_rate=0.03, epsilon=0.001, baseline=True):
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
        self.feature_list = []   # 坐标列表
        self.trajectory = []    # 经验
        self.gamma = gamma  # 折扣
        self.learning_rate = learning_rate  # 学习率
        self.epsilon = epsilon  # 探索
        self.layers = layers

    def decide(self, observation):
        '''
        决策函数
        observation  状态观测值 (位置 速度 时间)
        '''
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        else:
            qs = [self.get_q(observation[:2], action) for action in range(self.action_n)]
            return np.argmax(qs)

    def learn(self, observation, action, reward, done):
        '''
        学习函数
        observation  状态观测值 位置 速度
        action      动作
        reward      奖励
        done        是否完成
        next_action 下一动作
        '''
        q = self.get_q(observation[:2], action)

        self.trajectory.append((observation, action, reward, q))
        self.feature_list.append(self.encode(observation[:2], action))
        # 如果程序结束 储存数据
        if done:
            fertures_array = np.array(self.feature_list)
            df = pd.DataFrame(self.trajectory, columns=['observation', 'action', 'reward', 'q'])
            df['discount'] = self.gamma ** df.index.to_series()
            df['discounted_reward'] = df['discount'] * df['reward']
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()
            df['derivative'] = df['discounted_return'] - df['q']
            df['discounted_derivative'] = (df['discount'] * df['derivative'])
            # G = df['discounted_return']
            df['psi'] = df['discounted_return']

            # x = np.stack(df['observation'])
            # # 更新基线
            # # np.newaxis的作用就是在这一位置增加一个维度 例如一维向量变二维矩阵
            # # 基线的学习目标是 未来回报的贴现值
            # if self.baseline:
            #     df['baseline'] = self.baseline_net.predict(x)
            #     df['psi'] -= (df['baseline'] * df['discount'])
            #     df['return'] = df['discounted_return'] / df['discount']
            #     y = df['return'].values[:, np.newaxis]

            # 策略学习
            loss = df['discounted_derivative']
            for i in range(self.layers):
                self.w[fertures_array[:, i]] += self.learning_rate * loss
            # 为下一回合初始化经验列表
            self.trajectory = []
            self.feature_list = []


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


def play_learn_at_done(env, agent, train=False, render=False, collect_ex=False):
    '''
    智能体环境交互逻辑 在回合运行结束时学习
    env       环境
    agent     智能体
    train     是否训练
    render    是否显示
    返回期望收益
    '''
    observation = env.reset()
    episode_reward = 0.
    done = False
    while not done:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, done)
        observation = next_observation
    return episode_reward, []


def train(env, agent, play_fun):
    # 训练
    env.continuous_time = False
    episodes = 300
    episode_rewards = []
    for episode in range(episodes):
        episode_reward, _ = play_fun(env, agent, train=True)
        episode_rewards.append(episode_reward)
        print("{}/{} sum_w:{:.2f} score:{:.2f}".format(episode, episodes, np.sum(np.sum(agent.w)), episode_reward))
    plt.plot(episode_rewards)
    plt.show()

    # 测试
    env.continuous_time = False
    agent.epsilon = 0.0
    episodes = 100
    sum_rewards = 0.0
    for episode in range(episodes):
        episode_reward, _ = play_fun(env, agent, train=False)
        sum_rewards += episode_reward
        # print("train {}/{}".format(episode + 1, episodes))

    print('平均回合奖励 = {} / {} = {}'.format(sum_rewards, episodes, sum_rewards / episodes))


def train_difference_strayegy_orderly(env, learn_agent, behavior_agent, play_fun):
    '''
    异策略有序训练
    '''
    # 训练
    env.continuous_time = False
    episodes = 300
    episode_rewards = []
    for episode in range(episodes):
        _, trajectory = play_fun(env, behavior_agent, train=False, collect_ex=True)
        learn_agent.learn(trajectory)
        episode_reward, _ = play_fun(env, learn_agent, train=False)
        episode_rewards.append(episode_reward)
        # print("{}/{} sum_w:{:.2f} score:{:.2f}".format(episode, episodes, np.sum(np.sum(learn_agent.w)), episode_reward))
    plt.plot(episode_rewards)
    plt.show()

    # 测试
    env.continuous_time = False
    learn_agent.epsilon = 0.0
    episodes = 100
    sum_rewards = 0.0
    for episode in range(episodes):
        episode_reward, _ = play_fun(env, learn_agent, train=False)
        sum_rewards += episode_reward
        # print("train {}/{}".format(episode + 1, episodes))

    print('平均回合奖励 = {} / {} = {}'.format(sum_rewards, episodes, sum_rewards / episodes))


def train_difference_strayegy_sample(env, learn_agent, behavior_agent, play_fun):
    '''
    异策略随机抽样训练
    '''
    # 训练
    env.continuous_time = False
    episodes = 300
    episode_rewards = []
    trajectorys = []
    for episode in range(episodes):
        _, trajectory = play_fun(env, behavior_agent, train=False, collect_ex=True)
        trajectorys += trajectory

    seed(0)
    sample_n = int(len(trajectorys) / episodes)
    for episode in range(episodes):
        trajectory = sample(trajectorys, sample_n)
        learn_agent.learn(trajectory)
        episode_reward, _ = play_fun(env, learn_agent, train=False)
        episode_rewards.append(episode_reward)
        # print("{}/{} sum_w:{:.2f} score:{:.2f}".format(episode, episodes, np.sum(np.sum(learn_agent.w)), episode_reward))
    plt.plot(episode_rewards)
    plt.show()

    # 测试
    env.continuous_time = False
    learn_agent.epsilon = 0.0
    episodes = 100
    sum_rewards = 0.0
    for episode in range(episodes):
        episode_reward, _ = play_fun(env, learn_agent, train=False)
        sum_rewards += episode_reward
        # print("{}/{} sum_w:{:.2f} score:{:.2f}".format(episode, episodes, np.sum(np.sum(learn_agent.w)), episode_reward))
        # print("train {}/{}".format(episode + 1, episodes))

    print('平均回合奖励 = {} / {} = {}'.format(sum_rewards, episodes, sum_rewards / episodes))


def main():
    np.random.seed(0)
    env = MountainCarEnv()
    env.seed(0)
    env.continuous_time = False

    # sarsa_agent = SARSAAgent(env)
    # print(">>>>>>>>>>>>>>>>>>>>SARSA 算法<<<<<<<<<<<<<<<<<<<<<")
    # train(env, sarsa_agent, play_sarsa)

    vpg_agent = VPGAgent(env, gamma=1.0)
    print(">>>>>>>>>>>>>>>>>>>>VPG 算法<<<<<<<<<<<<<<<<<<<<<")
    train(env, vpg_agent, play_learn_at_done)
    return 0
    # lambda_agent = SARSALambdaAgent(env)
    # print(">>>>>>>>>>>>>>>>>>>>SARSA(λ) 算法<<<<<<<<<<<<<<<<<<<<<")
    # train(env, lambda_agent, play_sarsa)

    random_agent = RandomAgent(env)
    diff_agent_orderly = DiffAgent(env, learning_rate=0.03)
    diff_agent_sample = DiffAgent(env, learning_rate=0.03)
    # sarsa_agent.epsilon = 0.3
    print(">>>>>>>>>>>>>>>>>>>>SARSA离线学习 按序更新 算法<<<<<<<<<<<<<<<<<<<<<")
    train_difference_strayegy_orderly(env, diff_agent_orderly, random_agent, play_sarsa)
    print(">>>>>>>>>>>>>>>>>>>>SARSA离线学习 随机抽样 算法<<<<<<<<<<<<<<<<<<<<<")
    train_difference_strayegy_sample(env, diff_agent_sample, random_agent, play_sarsa)
    # train(env, random_agent, play_sarsa)

    # sarsa_grid_agent = SARSAAgent_grid(env, dim_size_arr=(16, 16), learning_rate=0.05, gamma=0.97)
    # print(">>>>>>>>>>>>>>>>>>>>SARSA_grid 算法<<<<<<<<<<<<<<<<<<<<<")
    # train(env, sarsa_grid_agent, play_sarsa)
    # print("program__end")


if __name__ == "__main__":
    main()
