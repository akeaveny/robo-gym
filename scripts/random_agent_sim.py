import gym
from gym.wrappers import TimeLimit
from gym.wrappers import FlattenObservation

import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from robo_gym.wrappers.flatten_action_space import FlattenAction

target_machine_ip = 'localhost' # or other machine 'xxx.xxx.xxx.xxx'

import numpy as np
import pprint

#####################
### robo-gym
#####################

# robot = 'NoObstacleNavigationMir100Sim-v0'
# robot = 'EndEffectorPositioningUR10Sim-v0'
#
# env = gym.make(robot, ip=target_machine_ip, gui=True)
# env = ExceptionHandling(env)

#####################
### UWRT
#####################
import config

robot = 'UWRTArmSim-v0'
# robot = 'Gen3Lite2FArmEnv-v0'

env = FlattenAction(FlattenObservation(gym.make(robot, ip=target_machine_ip, gui=True,
                                                key_position=config.KEY_POSITION,
                                                key_orientation=config.KEY_ORIENTATION,
                                                max_steps=config.MAX_STEPS_PER_EPISODE)))
env = ExceptionHandling(env).env

#####################
#####################
pp = pprint.PrettyPrinter()
num_episodes = 10

for episode in range(num_episodes):
    print()
    print(f'Episode #{episode} Starting!')
    print()

    done = False
    initial_observation = env.reset()
    print('Initial Observation:')
    pp.pprint(initial_observation)
    steps = 0
    while not done:
        steps += 1
        # random step in the environment
        action = env.action_space.sample()
        observation, reward, done, info = env.step(np.array(action)) # need flatten action

        # print()
        # print('Action:')
        # pp.pprint(action)
        # print('Observation:')
        # pp.pprint(observation)
        # print('Info:')
        # pp.pprint(info)
        # print('Reward:')
        # pp.pprint(reward)

        if done:
            print()
            print(f'Episode #{episode} finished after {steps} steps!')
            print(f'Episode #{episode} exit condition was {info["sim"]["end_condition"]}')
            print()

            break

