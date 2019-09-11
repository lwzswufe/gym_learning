# coding:utf-8
from abc import abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym


class BaseAgent(object):
    '''
    智能体类
    '''
    @abstractmethod
    def decide(self, state):
        '''
        决策函数
        state 当前状态
        '''

    @abstractmethod
    def learn(self, state, action, reward, next_state, done, next_action):
        '''
        学习函数
        state       状态
        action      动作
        reward      奖励
        next_state  下一状态
        done        是否完成
        next_action 下一动作
        '''


class SARSAAgent(BaseAgent):
    '''
    SARSA 算法
    智能体类
    '''
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state):
        '''
        决策函数
        state 当前状态 0-499
        返回 动作
        '''
        if np.random.uniform() > self.epsilon:
            # 最优策略
            action = self.q[state].argmax()
        else:  # 随机探索
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done, next_action):
        '''
        学习函数
        state       状态
        action      动作
        reward      奖励
        next_state  下一状态
        done        是否完成
        next_action 下一动作
        '''
        # 计算回报估计值
        u = reward + self.gamma * self.q[next_state, next_action] * (1. - done)
        # 更新q(S, A)
        self.q[state, action] += self.learning_rate * (u - self.q[state, action])


def play_sarsa(env, agent, train=False, render=False):
    '''
    智能体环境交互逻辑
    env       环境
    agent     智能体
    train     是否训练
    render    是否显示
    返回期望收益
    '''
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        next_action = agent.decide(next_observation)  # 终止状态时此步无意义
        if train:
            agent.learn(observation, action, reward, next_observation, done, next_action)
        if done:
            break
        observation, action = next_observation, next_action
    return episode_reward


class ExpectedSARSAAgent(BaseAgent):
    '''
    期望 SARSA
    '''
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q = np.zeros((env.observation_space.n, env.action_space.n))
        self.action_n = env.action_space.n

    def decide(self, state):
        '''
        决策函数
        state 当前状态 0-499
        返回 动作
        '''
        if np.random.uniform() > self.epsilon:
            # 最优化策略
            action = self.q[state].argmax()
        else:  # 随机策略
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done):
        '''
        学习函数
        state       状态
        action      动作
        reward      奖励
        next_state  下一状态
        done        是否完成
        next_action 下一动作
        '''
        v = (self.q[next_state].sum() * self.epsilon + self.q[next_state].max() * (1. - self.epsilon))
        u = reward + self.gamma * v * (1. - done)
        self.q[state, action] += self.learning_rate * (u - self.q[state, action])


def play_qlearning(env, agent, train=False, render=False):
    '''
    智能体环境交互逻辑
    env       环境
    agent     智能体
    train     是否训练
    render    是否显示
    返回期望收益
    '''
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, next_observation, done)
        if done:
            break
        observation = next_observation
    return episode_reward


class QLearningAgent:
    '''
    Q学习
    '''
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state):
        '''
        决策函数
        state 当前状态 0-499
        返回 动作
        '''
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done):
        '''
        学习函数
        state       状态
        action      动作
        reward      奖励
        next_state  下一状态
        done        是否完成
        next_action 下一动作
        '''
        u = reward + self.gamma * self.q[next_state].max() * (1. - done)
        self.q[state, action] += self.learning_rate * (u - self.q[state, action])


class DoubleQLearningAgent(QLearningAgent):
    def __init__(self, env, gamma=0.9, learning_rate=0.15, epsilon=.01):
        super().__init__(env, gamma, learning_rate, epsilon)
        self.q1 = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state):
        '''
        决策函数
        state 当前状态 0-499
        返回 动作
        '''
        if np.random.uniform() > self.epsilon:
            action = (self.q + self.q1)[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done):
        '''
        学习函数
        state       状态
        action      动作
        reward      奖励
        next_state  下一状态
        done        是否完成
        next_action 下一动作
        '''
        # 等概率随机选择q0 q1 进行训练
        if np.random.randint(2):
            self.q, self.q1 = self.q1, self.q
        a = self.q[next_state].argmax()
        u = reward + self.gamma * self.q1[next_state, a] * (1. - done)
        self.q[state, action] += self.learning_rate * (u - self.q[state, action])


class SARSALambdaAgent(SARSAAgent):
    '''
    资格迹
    '''
    def __init__(self, env, lambd=0.5, beta=1., gamma=0.9, learning_rate=0.1, epsilon=.01):
        super().__init__(env, gamma=gamma, learning_rate=learning_rate, epsilon=epsilon)
        self.lambd = lambd
        self.beta = beta
        # 资格迹矩阵
        self.e = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self, state, action, reward, next_state, done, next_action):
        '''
        学习函数
        state       状态
        action      动作
        reward      奖励
        next_state  下一状态
        done        是否完成
        next_action 下一动作
        '''
        # 更新资格迹
        self.e *= (self.lambd * self.gamma)
        self.e[state, action] = 1. + self.beta * self.e[state, action]

        # 更新价值
        u = reward + self.gamma * self.q[next_state, next_action] * (1. - done)
        self.q += self.learning_rate * self.e * (u - self.q[state, action])
        if done:
            self.e *= 0.


def main():
    np.random.seed(0)
    env = gym.make('Taxi-v2')
    env.seed(0)
    print('观察空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print('状态数量 = {}'.format(env.observation_space.n))
    print('动作数量 = {}'.format(env.action_space.n))

    state = env.reset()
    taxirow, taxicol, passloc, destidx = env.unwrapped.decode(state)
    print(taxirow, taxicol, passloc, destidx)
    print('的士位置 = {}'.format((taxirow, taxicol)))
    print('乘客位置 = {}'.format(env.unwrapped.locs[passloc]))
    print('目标位置 = {}'.format(env.unwrapped.locs[destidx]))
    env.render()

    env.step(0)
    env.render()

    agent = SARSAAgent(env)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>开始训练SARSA<<<<<<<<<<<<<<<<<<<<<<<<<<")
    episodes = 5000
    episode_rewards = []
    for episode in range(episodes):
        episode_reward = play_sarsa(env, agent, train=True)
        episode_rewards.append(episode_reward)

    plt.plot(episode_rewards)

    # 测试
    agent.epsilon = 0.  # 取消探索

    episode_rewards = [play_sarsa(env, agent) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))

    pd.DataFrame(agent.q)
    policy = np.eye(agent.action_n)[agent.q.argmax(axis=-1)]
    pd.DataFrame(policy)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>开始训练期望SARSA<<<<<<<<<<<<<<<<<<<<<<<<<<")
    agent = ExpectedSARSAAgent(env)

    # 训练
    episodes = 5000
    episode_rewards = []
    for episode in range(episodes):
        episode_reward = play_qlearning(env, agent, train=True)
        episode_rewards.append(episode_reward)

    plt.plot(episode_rewards)

    # 测试
    agent.epsilon = 0.  # 取消探索

    episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>开始训练Q_Learning<<<<<<<<<<<<<<<<<<<<<<<<<<")
    agent = QLearningAgent(env)

    # 训练
    episodes = 5000
    episode_rewards = []
    for episode in range(episodes):
        episode_reward = play_qlearning(env, agent, train=True)
        episode_rewards.append(episode_reward)

    plt.plot(episode_rewards)

    # 测试
    agent.epsilon = 0.  # 取消探索

    episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>开始训练DoubleQ_Learning<<<<<<<<<<<<<<<<<<<<<<<<<<")
    agent = DoubleQLearningAgent(env)

    # 训练
    episodes = 5000
    episode_rewards = []
    for episode in range(episodes):
        episode_reward = play_qlearning(env, agent, train=True)
        episode_rewards.append(episode_reward)

    plt.plot(episode_rewards)

    # 测试
    agent.epsilon = 0.  # 取消探索

    episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>开始训练TD_lambda<<<<<<<<<<<<<<<<<<<<<<<<<<")
    agent = SARSALambdaAgent(env)

    # 训练
    episodes = 5000
    episode_rewards = []
    for episode in range(episodes):
        episode_reward = play_sarsa(env, agent, train=True)
        episode_rewards.append(episode_reward)

    plt.plot(episode_rewards)

    # 测试
    agent.epsilon = 0.  # 取消探索

    episode_rewards = [play_sarsa(env, agent, train=False) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))


if __name__ == "__main__":
    main()
