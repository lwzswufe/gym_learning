# author='lwz'
# coding:utf-8

import random
import numpy as np


def sampling(a):
    a = [1, 2, 2, 3]
    np.argsort(random.sample(a, 4))
