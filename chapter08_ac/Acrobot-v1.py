import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import gym
import tensorflow.compat.v2 as tf
from tensorflow import keras


class QActorCriticAgent:
    '''
    用简单的执行者评论家算法寻找最优策略
    '''
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.discount = 1.
        # 创建执行者网络
        self.actor_net = self.build_network(output_size=self.action_n,
                                            output_activation=tf.nn.softmax,
                                            loss=tf.losses.categorical_crossentropy,
                                            **actor_kwargs)
        # 创建评论者网络
        self.critic_net = self.build_network(output_size=self.action_n,
                                             **critic_kwargs)

    def build_network(self, hidden_sizes, output_size, input_size=None,
                      activation=tf.nn.relu, output_activation=None,
                      loss=tf.losses.mse, learning_rate=0.01):
        '''
        创建网络
        '''
        model = keras.Sequential()
        for idx, hidden_size in enumerate(hidden_sizes):
            kwargs = {}
            if idx == 0 and input_size is not None:
                kwargs['input_shape'] = (input_size,)
            model.add(keras.layers.Dense(units=hidden_size,
                      activation=activation, **kwargs))
        model.add(keras.layers.Dense(units=output_size,
                  activation=output_activation))
        optimizer = tf.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def decide(self, observation):
        '''
        决策函数
        observation 观测值
        return action 动作
        '''
        probs = self.actor_net.predict(observation[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, next_observation, done):
        '''
        训练网络
        学习函数
        observation  状态观测值 位置 速度
        action      动作
        reward      奖励
        next_observation  下一状态 观测值
        done        是否完成
        '''
        # 训练执行者网络
        x = observation[np.newaxis]
        u = self.critic_net.predict(x)
        q = u[0, action]
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)[0, action]
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.))
            loss_tensor = -self.discount * q * logpi_tensor
        grad_tensors = tape.gradient(loss_tensor, self.actor_net.variables)
        self.actor_net.optimizer.apply_gradients(zip(
                grad_tensors, self.actor_net.variables))

        # 训练评论者网络
        q = self.critic_net.predict(next_observation[np.newaxis])[0, action]
        u[0, action] = reward + (1. - done) * self.gamma * q
        self.critic_net.fit(x, u, verbose=0)

        if done:
            self.discount = 1.
        else:
            self.discount *= self.gamma


class AdvantageActorCriticAgent(QActorCriticAgent):
    '''
    优势执行者/评论者算法
    '''
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.discount = 1.
        # 创建执行者网络
        self.actor_net = self.build_network(output_size=self.action_n,
                                            output_activation=tf.nn.softmax,
                                            loss=tf.losses.categorical_crossentropy,
                                            **actor_kwargs)
        # 创建评论者网络
        self.critic_net = self.build_network(output_size=1, **critic_kwargs)

    def learn(self, observation, action, reward, next_observation, done):
        '''
        训练网络
        学习函数
        observation  状态观测值 位置 速度
        action      动作
        reward      奖励
        next_observation  下一状态 观测值
        done        是否完成
        '''
        x = observation[np.newaxis]
        u = reward + (1. - done) * self.gamma * self.critic_net.predict(next_observation[np.newaxis])
        td_error = u - self.critic_net.predict(x)
        y = self.discount * td_error * np.eye(self.action_n)[np.newaxis, action]
        self.actor_net.fit(x, y, verbose=0)
        self.critic_net.fit(x, u, verbose=0)

        if done:
            self.discount = 1.
        else:
            self.discount *= self.gamma


class ElibilityTraceActorCriticAgent(QActorCriticAgent):
    '''
    训练网络
    学习函数
    observation  状态观测值 位置 速度
    action      动作
    reward      奖励
    next_observation  下一状态 观测值
    done        是否完成
    '''
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99,
                 actor_lambda=0.9, critic_lambda=0.9):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.actor_lambda = actor_lambda
        self.critic_lambda = critic_lambda
        self.gamma = gamma
        self.discount = 1.
        # 创建执行者网络
        self.actor_net = self.build_network(input_size=observation_dim,
                                            output_size=self.action_n,
                                            output_activation=tf.nn.softmax,
                                            **actor_kwargs)
        # 创建评论者网络
        self.critic_net = self.build_network(input_size=observation_dim,
                                             output_size=1, **critic_kwargs)
        # 执行者资格迹
        self.actor_traces = [np.zeros_like(weight) for weight in self.actor_net.get_weights()]
        # 创建者资格迹
        self.critic_traces = [np.zeros_like(weight) for weight in self.critic_net.get_weights()]

    def learn(self, observation, action, reward, next_observation, done):
        '''
        训练网络
        学习函数
        observation  状态观测值 位置 速度
        action      动作
        reward      奖励
        next_observation  下一状态 观测值
        done        是否完成
        '''
        q = self.critic_net.predict(observation[np.newaxis])[0, 0]
        u = reward + (1. - done) * self.gamma * self.critic_net.predict(next_observation[np.newaxis])[0, 0]
        td_error = u - q
        # 训练执行者网络
        x_tensor = tf.convert_to_tensor(observation[np.newaxis], dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.))
            logpi_pick_tensor = logpi_tensor[0, action]
        grad_tensors = tape.gradient(logpi_pick_tensor, self.actor_net.variables)
        # 执行者资格迹
        self.actor_traces = [self.gamma * self.actor_lambda * trace +
                             self.discount * grad.numpy() for trace, grad in
                             zip(self.actor_traces, grad_tensors)]
        actor_grads = [tf.convert_to_tensor(-td_error * trace,
                       dtype=tf.float32) for trace in self.actor_traces]
        # 计算梯度
        actor_grads_and_vars = tuple(zip(actor_grads, self.actor_net.variables))
        # 输入梯度更新网络
        self.actor_net.optimizer.apply_gradients(actor_grads_and_vars)

        # 训练评论者网络
        with tf.GradientTape() as tape:
            v_tensor = self.critic_net(x_tensor)
        grad_tensors = tape.gradient(v_tensor, self.critic_net.variables)
        self.critic_traces = [self.gamma * self.critic_lambda * trace +
                              self.discount * grad.numpy() for trace, grad in
                              zip(self.critic_traces, grad_tensors)]
        critic_grads = [tf.convert_to_tensor(-td_error * trace,
                        dtype=tf.float32) for trace in self.critic_traces]
        # 计算梯度
        critic_grads_and_vars = tuple(zip(critic_grads, self.critic_net.variables))
        # 输入梯度更新网络
        self.critic_net.optimizer.apply_gradients(critic_grads_and_vars)

        if done:
            # 下一回合重置资格迹
            self.actor_traces = [np.zeros_like(weight) for weight in self.actor_net.get_weights()]
            self.critic_traces = [np.zeros_like(weight) for weight in self.critic_net.get_weights()]
            # 为下一回合重置累积折扣
            self.discount = 1.
        else:
            self.discount *= self.gamma


class OffPACAgent:
    '''
    异策算法
    '''
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.discount = 1.
        self.critic_learning_rate = critic_kwargs['learning_rate']
        # 创建执行者网络
        self.actor_net = self.build_network(output_size=self.action_n,
                                            output_activation=tf.nn.softmax,
                                            **actor_kwargs)
        # 创建评论者网络
        self.critic_net = self.build_network(output_size=self.action_n,
                                             **critic_kwargs)

    def build_network(self, hidden_sizes, output_size,
                      activation=tf.nn.relu, output_activation=None,
                      loss=tf.losses.mse, learning_rate=0.01):
        '''
        创建网络
        '''
        model = keras.Sequential()
        for idx, hidden_size in enumerate(hidden_sizes):
            model.add(keras.layers.Dense(units=hidden_size, activation=activation))
        model.add(keras.layers.Dense(units=output_size, activation=output_activation))
        optimizer = tf.optimizers.SGD(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def decide(self, observation):
        probs = self.actor_net.predict(observation[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, behavior, reward, next_observation, done):
        observations = np.float32(observation[np.newaxis])
        # 用于训练评论者
        pi = self.actor_net(observations)[0, action]
        # 用于训练执行者
        q = self.critic_net(observations)[0, action]

        # 训练执行者
        x_tensor = tf.convert_to_tensor(observations, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)
            loss_tensor = -self.discount * q / behavior * pi_tensor[0, action]
        grad_tensors = tape.gradient(loss_tensor, self.actor_net.variables)
        self.actor_net.optimizer.apply_gradients(zip(grad_tensors, self.actor_net.variables))

        # 训练评论者
        next_q = self.critic_net.predict(next_observation[np.newaxis])[0, action]
        u = reward + self.gamma * (1. - done) * next_q
        u_tensor = tf.convert_to_tensor(u, dtype=tf.float32)
        with tf.GradientTape() as tape:
            q_tensor = self.critic_net(x_tensor)
            mse_tensor = tf.losses.MSE(u_tensor, q_tensor)
            loss_tensor = pi / behavior * mse_tensor
        grad_tensors = tape.gradient(loss_tensor, self.critic_net.variables)
        self.critic_net.optimizer.apply_gradients(zip(grad_tensors, self.critic_net.variables))

        if done:
            self.discount = 1.
        else:
            self.discount *= self.gamma


class RandomAgent:
    '''
    随机策略智能体
    '''
    def __init__(self, env):
        self.action_n = env.action_space.n

    def decide(self, observation):
        action = np.random.choice(self.action_n)
        behavior = 1. / self.action_n
        return action, behavior


def play_qlearning(env, agent, train=False, render=False):
    '''
    智能体环境交互函数
    env       环境
    agent     智能体
    train     是否训练
    render    是否显示
    返回期望收益
    '''
    episode_reward = 0
    observation = env.reset()
    step = 0
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
        step += 1
        observation = next_observation
    return episode_reward


def play_qlearning_off_strategy(env, agent, behavior_agent):
    '''
    离线策略智能体环境交互函数
    env       环境
    agent     智能体
    behavior_agent 行为智能体
    '''
    episodes = 80
    episode_rewards = []
    for episode in range(episodes):
        observation = env.reset()
        episode_reward = 0.
        while True:
            action, behavior = behavior_agent.decide(observation)
            next_observation, reward, done, _ = env.step(action)
            episode_reward += reward
            agent.learn(observation, action, behavior, reward, next_observation, done)
            if done:
                break
            observation = next_observation

        # 跟踪监控
        episode_reward = play_qlearning(env, agent)
        episode_rewards.append(episode_reward)

    plt.plot(episode_rewards)

    # 测试
    episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))


def show(env, agent, play_fun):
    '''
    显示策略效果
    env      环境
    agent    智能体
    play_fun 交互函数
    '''
    # 训练
    episodes = 100
    episode_rewards = []
    for episode in range(episodes):
        episode_reward = play_qlearning(env, agent, train=True)
        episode_rewards.append(episode_reward)
    plt.plot(episode_rewards)
    plt.show()
    # 测试
    episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))


def main():
    np.random.seed(0)
    tf.random.set_seed(0)
    env = gym.make('Acrobot-v1')
    env.seed(0)
    print(">>>>>>>>>>>>>>>>>>>简单的执行者评论家算法<<<<<<<<<<<<<<<<<<")
    actor_kwargs = {'hidden_sizes': [20, ], 'learning_rate': 0.0005}
    critic_kwargs = {'hidden_sizes': [20, ], 'learning_rate': 0.0005}
    agent = QActorCriticAgent(env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs)
    show(env, agent, play_qlearning)

    print(">>>>>>>>>>>>>>>>>>>优势执行者/评论者算法<<<<<<<<<<<<<<<<<<")
    actor_kwargs = {'hidden_sizes': [20, ], 'learning_rate': 0.0001}
    critic_kwargs = {'hidden_sizes': [20, ], 'learning_rate': 0.0002}
    agent = AdvantageActorCriticAgent(env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs)
    show(env, agent, play_qlearning)

    print(">>>>>>>>>>>>>>>>>>>带资格迹的执行者/评论者算法<<<<<<<<<<<<<<<<<<")
    actor_kwargs = {'hidden_sizes': [20, ], 'learning_rate': 0.0001}
    critic_kwargs = {'hidden_sizes': [20, ], 'learning_rate': 0.0001}
    agent = ElibilityTraceActorCriticAgent(env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs)

    print(">>>>>>>>>>>>>>>>>>>重要性采样异策算法的执行者/评论者算法<<<<<<<<<<<<<<<<<<")
    actor_kwargs = {'hidden_sizes': [20, ], 'learning_rate': 0.0005}
    critic_kwargs = {'hidden_sizes': [20, ], 'learning_rate': 0.0005}
    agent = OffPACAgent(env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs)
    behavior_agent = RandomAgent(env)
    play_qlearning_off_strategy(env, agent, behavior_agent)


if __name__ == "__main__":
    main()
