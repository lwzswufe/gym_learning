import numpy as np
import pandas as pd
import gym


def introduction():
    '''
    Gym环境简介
    '''
    space_names = ['观测空间', '动作空间', '奖励范围', '最大步数']
    df = pd.DataFrame(columns=space_names)

    env_specs = gym.envs.registry.all()
    for env_spec in env_specs:
        env_id = env_spec.id
        try:
            env = gym.make(env_id)
            observation_space = env.observation_space
            action_space = env.action_space
            reward_range = env.reward_range
            max_episode_steps = None
            if isinstance(env, gym.wrappers.time_limit.TimeLimit):
                max_episode_steps = env._max_episode_steps
            df.loc[env_id] = [observation_space, action_space, reward_range, max_episode_steps]
        except:
            pass

    with pd.option_context('display.max_rows', None):
        print(df)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


# 智能体：一个根据指定确定性策略决定动作并且不学习的智能体
class BespokeAgent:
    def __init__(self, env):
        pass

    def decide(self, observation):  # 决策
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action  # 返回动作

    def learn(self, *args):  # 学习
        pass


def play_montecarlo(env, agent, render=False, train=False):
    '''
    智能体与环境交互
    env     环境
    agent   智能体
    render  是否显示 
    train   是否训练
    '''
    episode_reward = 0.   # 记录回合总奖励，初始化为0
    observation = env.reset()   # 重置游戏环境，开始新回合
    while True:   # 不断循环，直到回合结束
        if render:  # 判断是否显示
            env.render()  # 显示图形界面，图形界面可以用 env.close() 语句关闭
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)  # 执行动作
        episode_reward += reward  # 收集回合奖励
        if train:  # 判断是否训练智能体
            agent.learn(observation, action, reward, done)  # 学习
        if done:  # 回合结束，跳出循环
            break
        observation = next_observation
    return episode_reward  # 返回回合总奖励


def main():
    '''
    主程序
    '''
    env = gym.make('MountainCar-v0')
    print('观测空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print('观测范围 = {} ~ {}'.format(env.observation_space.low, env.observation_space.high))
    print('动作数 = {}'.format(env.action_space.n))

    agent = BespokeAgent(env)
    env.seed(0)  # 设置随机数种子,只是为了让结果可以精确复现,一般情况下可删去
    episode_reward = play_montecarlo(env, agent, render=True)
    print('回合奖励 = {}'.format(episode_reward))
    env.close()  # 此语句可关闭图形界面

    episode_rewards = [play_montecarlo(env, agent) for _ in range(100)]
    print('平均回合奖励 = {}'.format(np.mean(episode_rewards)))


if __name__ == "__main__":
    introduction()
    main()
