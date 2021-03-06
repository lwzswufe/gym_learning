import logging
import numpy
import random
from gym import spaces
import gym
from gym.utils import seeding
from gym.envs.classic_control import rendering


logger = logging.getLogger(__name__)


class GridMaze(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 3
    }

    def __init__(self, width=12, height=12):
        self.width_cell = width
        self.height_cell = height
        grid_num = width * height
        self.grid_num = grid_num
        self.q_value = numpy.zeros(grid_num)
        blank_width = 50
        x_max = 600 + blank_width
        grid_width = int((x_max - blank_width) / width)
        grid_height = grid_width
        self.grid_width = grid_width

        x_max = int(blank_width + grid_width * width)
        y_max = int(blank_width + grid_height * height)
        self.states = list(range(grid_num))  # 状态空间
        self.x = [blank_width + (0.5 + (i % width)) * grid_width for i in range(grid_num)]
        # 中心坐标参数
        self.y = [y_max - (0.5 + numpy.floor(i / width)) * grid_height for i in range(grid_num)]
        # 中心坐标参数

        if width != 5 and height != 5:
            is_random = True
        else:
            is_random = False

        terminate_states = []
        self.maze = numpy.zeros(height * width)

        if is_random:
            self.wall_states = random.sample(range(grid_num), int(grid_num / 4))
            terminate_states.append(self.wall_states.pop(-1))
        else:
            self.wall_states = [3, 8, 10, 11, 22, 23, 24]
            terminate_states = [14]

        for state in terminate_states:
            self.maze[state] = 1

        for state in self.wall_states:
            self.maze[state] = -1

        self.terminate_states = dict()  # 终止状态为字典格式
        for key in terminate_states:
            self.terminate_states[key] = 1

        self.actions = [0, 1, 2, 3]
        # n上0  e右1  s下2  w左3

        self.rewards = dict()         # 回报的数据结构为字典
        for i in self.terminate_states:
            self.rewards[i] = 1

        # 状态转移的数据格式为字典
        self.t = numpy.zeros([grid_num, 4])
        for i in range(grid_num):
            if i + width in self.wall_states:
                self.t[i, 0] = i
            elif i + width >= grid_num:
                self.t[i, 0] = i
            else:
                self.t[i, 0] = i + width

            if i + 1 in self.wall_states:
                self.t[i, 1] = i
            elif (i + 1) % width == 0:
                self.t[i, 1] = i
            else:
                self.t[i, 1] = i + 1

            if i - width in self.wall_states:
                self.t[i, 2] = i
            elif i - width < 0:
                self.t[i, 2] = i
            else:
                self.t[i, 2] = i - width

            if i - 1 in self.wall_states:
                self.t[i, 3] = i
            elif i % width == 0:
                self.t[i, 3] = i
            else:
                self.t[i, 3] = i - 1

        self.gamma = 0.8         # 折扣因子
        self.viewer = None
        self.state = None

        self.screen_height = y_max + blank_width
        self.screen_width = x_max + blank_width
        self.Lines = []  # 设置线

        for i in range(height + 1):
            Line = rendering.Line((blank_width, y_max - i * grid_height), (x_max, y_max - i * grid_height))
            Line.set_color(0, 0, 0)
            self.Lines.append(Line)

        for j in range(width + 1):
            Line = rendering.Line((blank_width + j * grid_width, y_max), (blank_width + j * grid_width, blank_width))
            Line.set_color(0, 0, 0)
            self.Lines.append(Line)

        self.Walls = []  # 设置墙

        for i in self.states:
            v = [(-grid_height/2, -grid_width/2), (-grid_height/2,  grid_width/2),
                 ( grid_height/2,  grid_width/2), ( grid_height/2, -grid_width/2)]
            wall = rendering.make_polygon(v)
            circletrans = rendering.Transform(translation=(self.x[i], self.y[i]))
            wall.add_attr(circletrans)

            if i in self.wall_states:
                wall.set_color(0, 0, 0)
            else:
                wall.set_color(0.5, 0.5, 0.5)

            self.Walls.append(wall)

        self.Doors = []  # 设置出口

        for i in self.terminate_states.keys():
            door = rendering.make_circle(grid_width/2)
            circletrans = rendering.Transform(translation=(self.x[i], self.y[i]))
            door.add_attr(circletrans)
            door.set_color(1, 0.5, 0.5)
            self.Doors.append(door)

    def update_Q_value(self, state=0, value=0):
        self.q_value[state] = value
        if value > 0:
            value = min(value, 1)
            r = 1
            g = 1 - value
            b = 0
        else:
            value = max(-1, value)
            g = 1
            r = 1 + value
            b = 0

        self.Walls[state].set_color(r, g, b)


    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions

    def getTerminate_states(self):
        return self.terminate_states

    def setAction(self,s):
        self.state=s

    def _step(self, action):
        # 系统当前状态
        state = self.state
        if state in self.terminate_states:  # 终止节点[6, 7, 8]
            return state, 0, True, {}
        elif state in self.wall_states:  # 终止节点[6, 7, 8]
            return state, 0, True, {}

        key = self.actions.index(action)  # 将状态和动作组成字典的键值

        # 状态转移
        next_state = int(self.t[state, key])
        self.state = next_state

        is_terminal = False

        if next_state in self.terminate_states:
            is_terminal = True

        if next_state not in self.rewards:
            r = -0.10
        else:
            r = self.rewards[next_state]

        return next_state, r, is_terminal, {}

    def _reset(self):
        self.state = self.states[random.randint(0, self.grid_num - 1)]
        while self.state in self.wall_states:
            self.state = self.states[random.randint(0, self.grid_num - 1)]

        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = int(self.screen_width)
        screen_height = int(self.screen_height)

        if self.viewer is None:

            self.viewer = rendering.Viewer(screen_width, screen_height)
            # 创建网格世界
            # 创建机器人
            robot = rendering.make_circle(self.grid_width/2)
            self.robotrans = rendering.Transform()
            robot.add_attr(self.robotrans)
            robot.set_color(0.8, 0.6, 0.4)

            for Line in self.Lines:
                self.viewer.add_geom(Line)

            for Wall in self.Walls:
                self.viewer.add_geom(Wall)

            for door in self.Doors:
                self.viewer.add_geom(door)

            self.viewer.add_geom(robot)

        if self.state is None:
            return None

        # 更新机器人坐标

        self.robotrans.set_translation(self.x[self.state], self.y[self.state])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
