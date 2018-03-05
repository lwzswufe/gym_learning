# author='lwz'
# coding:utf-8
'''
1 将文件 grid_maze.py 复制到gym/gym/envs/classic_control
2 将文件夹中打开__init__.py 在末尾加入：
    from gym.envs.classic_control.grid_maze import GridMaze
3 打开 gym/gym/envs/__init__.py  在末尾加入：
    register(
            id = 'GridMaze-v0',
            entry_point='gym.envs.classic_control:GridMaze',
            max_episode_steps = 200,
            reward_threshold = 100.0,
            )
'''
import gym, time, random


env = gym.make('GridMaze-v0')
env.reset()
actions = ['n', 'e', 's', 'w']
for i in range(500):
    time.sleep(0.05)
    action = actions[random.randint(0, 3)]
    next_state, r, is_terminal, _ = env.step(action)

    if is_terminal:
        env.reset()
        print("times: {} state:{} return:{}".format(i, next_state, r))
    else:
        print("times: {} state:{}".format(i, next_state))

    env.render()

env.close()  # 设置终止 否则会出现迭代错误