# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:13:29 2022

@author: yoda
"""

import gym
import numpy as np
import time
env = gym.make("MountainCar-v0")
env.reset()
EPISODES=100
SHOW_EVERY=1
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

done = False
for episode in range(EPISODES):
    action = 2
    new_state, reward, done, _ = env.step(action)
    print(reward, new_state)
  
    if episode % SHOW_EVERY == 0:
        env.render()
        time.sleep(2)