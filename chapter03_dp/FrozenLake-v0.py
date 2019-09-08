# coding:utf-8
import numpy as np
import gym

'''
冰面滑行 FrozenLake-v0
'''


def play_policy(env, policy, render=False):
    '''
    随机策略
    env    环境
    policy 策略 位置*4 的矩阵 表示在每一个点上选择不同方向的概率
    render 是否显示
    '''
    total_reward = 0.
    observation = env.reset()   # 当前位置
    while True:
        if render:
            env.render()  # 此行可显示
        action = np.random.choice(env.action_space.n, p=policy[observation])
        observation, reward, done, _ = env.step(action)
        total_reward += reward  # 统计回合奖励
        if done:  # 游戏结束
            break
    return total_reward


def improve_policy(env, v, policy, gamma=1.):
    '''
    对于每一个位置 使用局部最优值 改进策略
    env      环境
    v        状态价值数组
    policy   策略
    返回 策略是否已经被优化
    '''
    optimal = True
    for s in range(env.unwrapped.nS):
        q = v2q(env, v, s, gamma)
        a = np.argmax(q)
        if policy[s][a] != 1.:
            optimal = False
            policy[s] = 0.
            policy[s][a] = 1.
    return optimal


def v2q(env, v, s=None, gamma=1.):
    '''
    根据状态价值函数计算动作价值函数
    env      环境
    v        状态价值数组
    s=None   状态
    gamma=1. 时间折扣因子
    返回 q 状态Q值数组
    '''
    if s is not None:  # 针对单个状态求解
        q = np.zeros(env.unwrapped.nA)
        for a in range(env.unwrapped.nA):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                q[a] += prob * (reward + gamma * v[next_state] * (1. - done))
    else:  # 针对所有状态求解
        q = np.zeros((env.unwrapped.nS, env.unwrapped.nA))
        for s in range(env.unwrapped.nS):
            q[s] = v2q(env, v, s, gamma)
    return q


def evaluate_policy(env, policy, gamma=1., tolerant=1e-6):
    '''
    策略评估函数
    env      环境
    policy   策略 位置-方向矩阵 表示在每一个点上选择不同方向的概率
    tolerant  迭代条件
    返回 v 状态价值数组
    '''
    v = np.zeros(env.unwrapped.nS)  # 初始化状态价值函数
    while True:  # 循环
        delta = 0
        for s in range(env.unwrapped.nS):
            vs = sum(policy[s] * v2q(env, v, s, gamma))  # 更新状态价值函数
            delta = max(delta, abs(v[s]-vs))  # 更新最大误差
            v[s] = vs  # 更新状态价值函数
        if delta < tolerant:  # 查看是否满足迭代条件
            break
    return v


def iterate_policy(env, gamma=1., tolerant=1e-6):
    '''
    策略迭代
    初始化为任意一个策略
    '''
    policy = np.ones((env.unwrapped.nS, env.unwrapped.nA)) / env.unwrapped.nA
    while True:
        v = evaluate_policy(env, policy, gamma, tolerant)  # 策略评估
        if improve_policy(env, v, policy):  # 策略改进
            break
    return policy, v


def iterate_value(env, gamma=1, tolerant=1e-6):
    '''
    # 价值迭代略
    env      环境
    policy   策略
    tolerant  迭代条件
    返回 policy   策略  v 状态价值数组
    '''
    v = np.zeros(env.unwrapped.nS)  # 初始化
    while True:
        delta = 0
        for s in range(env.unwrapped.nS):
            vmax = max(v2q(env, v, s, gamma))  # 更新价值函数
            delta = max(delta, abs(v[s]-vmax))
            v[s] = vmax
        if delta < tolerant:  # 满足迭代需求
            break

    policy = np.zeros((env.unwrapped.nS, env.unwrapped.nA))  # 计算最优策略
    for s in range(env.unwrapped.nS):
        a = np.argmax(v2q(env, v, s, gamma))
        policy[s][a] = 1.
    return policy, v


def main():
    np.random.seed(0)
    env = gym.make('FrozenLake-v0')
    env.seed(0)
    print('观察空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print('观测空间大小 = {}'.format(env.unwrapped.nS))
    print('动作空间大小 = {}'.format(env.unwrapped.nA))
    env.unwrapped.P[14][2]  # 查看动力

    random_policy = np.ones((env.unwrapped.nS, env.unwrapped.nA)) / env.unwrapped.nA

    episode_rewards = [play_policy(env, random_policy) for _ in range(100)]
    print("随机策略 平均奖励：{}".format(np.mean(episode_rewards)))

    print('状态价值函数：')
    v_random = evaluate_policy(env, random_policy)
    print(v_random.reshape(4, 4))

    print('动作价值函数：')
    q_random = v2q(env, v_random)
    print(q_random)

    policy = random_policy.copy()
    optimal = improve_policy(env, v_random, policy)
    if optimal:
        print('无更新，最优策略为：')
    else:
        print('有更新，更新后的策略为：')
    print(policy)

    policy_pi, v_pi = iterate_policy(env)
    print('状态价值函数 =')
    print(v_pi.reshape(4, 4))
    print('最优策略 =')
    print(np.argmax(policy_pi, axis=1).reshape(4, 4))

    episode_rewards = [play_policy(env, policy_pi) for _ in range(100)]
    print("策略迭代 平均奖励：{}".format(np.mean(episode_rewards)))

    policy_vi, v_vi = iterate_value(env)
    print('状态价值函数 =')
    print(v_vi.reshape(4, 4))
    print('最优策略 =')
    print(np.argmax(policy_vi, axis=1).reshape(4, 4))


if __name__ == "__main__":
    main()
