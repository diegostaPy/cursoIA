# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:13:29 2022

@author: yoda
"""

import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

done = False

while not done:
    action = 2
    new_state, reward, done, _ = env.step(action)
    print(reward, new_state)
    env.render()