# author='lwz'
# coding:utf-8
import numpy as np
import gym, time, random


GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
TRAIN_TIMES = 10000

env = gym.make('GridMaze-v0')
state_num = env.env.grid_num
env.reset()
actions = env.env.getAction()
action_num = len(actions)
Q = np.zeros([state_num, len(actions)])
epsilon = INITIAL_EPSILON

for i in range(TRAIN_TIMES):
    time.sleep(0.01)
    state = env.env.state

    if random.random() < epsilon:
        action_id = random.randint(0, action_num - 1)
    else:
        action_id = np.argmax(Q[state, :])

    next_state, r, is_terminal, _ = env.step(actions[action_id])

    if is_terminal:
        Q[state, action_id] = r
        time.sleep(0.1)
        env.reset()
        print("times: {} state:{} return:{}".format(i, next_state, r))
    else:
        Q[state, action_id] = r + GAMMA * np.max(Q[next_state, :])
        env.env.update_Q_value(state, np.max(Q[state, :]))
        print("times: {} state:{}".format(i, next_state))

    env.render()
    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / TRAIN_TIMES

env.close()  # 设置终止 否则会出现迭代错误