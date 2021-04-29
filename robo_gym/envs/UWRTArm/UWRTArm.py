#!/usr/bin/env python3

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import copy
import numpy as np
from scipy.spatial.transform import Rotation as R

from urdfpy import URDF
import requests

import gym
from gym import spaces
from gym.utils import seeding

from robo_gym.utils import utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2

class UWRTArmEnv(gym.Env):
    """ UWRT ARM base environment """

    # UWRT URDF
    ARM_URDF = '/home/akeaveny/git/robo-gym/robo_gym/envs/UWRTArm/urdf/robot.urdf'
    ARM_URDF_FILE_NAME = 'robot.urdf'

    # Reward Constants
    GOAL_POSITION_DISTANCE_THRESHOLD = 1 / 1000  # 1 mm
    REWARD_MAX = 100
    reward_range = (-float('inf'), float(REWARD_MAX))

    @dataclass(frozen=True)
    class InitOptions:
        __slots__ = ['key_position', 'key_orientation', 'max_steps', 'is_val_env', 'is_keyboard_demo_env',  'tmp_dir']
        key_position: np.ndarray
        key_orientation: np.ndarray

        max_steps: int

        is_val_env: bool
        is_keyboard_demo_env: bool

        tmp_dir: tempfile.TemporaryDirectory

    def __init__(self, key_position, key_orientation, max_steps, is_val_env=False, is_keyboard_demo_env=False, rs_address=None, **kwargs):

        self.init_options = self.InitOptions(key_position=key_position, key_orientation=key_orientation, max_steps=max_steps,
                                             is_val_env=is_val_env, is_keyboard_demo_env=is_keyboard_demo_env,
                                             tmp_dir=tempfile.TemporaryDirectory())
        self.__initialize_gym()

        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")

    def __initialize_gym(self):
        arm_urdf = URDF.load(UWRTArmEnv.ARM_URDF)
        urdf_arm_joint_idxs = [1, 4, 9, 10, 11] # these joint idxs are used to initalize joint limits

        num_joints = len(urdf_arm_joint_idxs)
        num_actuated_joints = len(arm_urdf.actuated_joint_names)
        actuated_joints_names = arm_urdf.actuated_joint_names

        joint_limits = []
        for joint_idx in range(num_actuated_joints):
            if joint_idx in urdf_arm_joint_idxs:
                if arm_urdf.actuated_joints[joint_idx].joint_type == 'continuous': # wrist_rotate
                    joint_limits.append((-75 * np.pi / 180.0, 75 * np.pi / 180.0))
                else:
                    joint_limits.append((arm_urdf.actuated_joints[joint_idx].limit.lower,
                                         arm_urdf.actuated_joints[joint_idx].limit.upper))

        joint_vel_limits = []
        for joint_idx in range(num_actuated_joints):
            if joint_idx in urdf_arm_joint_idxs:
                # joint_vel_limits.append((-1, 1)) # set all joint limits to 1 m/s
                if arm_urdf.actuated_joints[joint_idx].joint_type == 'continuous': # wrist_rotate
                    joint_vel_limits.append((-0.5, 0.5))
                else:
                    joint_vel_limits.append((-1 * arm_urdf.actuated_joints[joint_idx].limit.velocity,
                                             arm_urdf.actuated_joints[joint_idx].limit.velocity))

        # All joint limit switch states are either NOT_TRIGGERED[0], LOWER_TRIGGERED[1], UPPER_TRIGGERED[2]
        # The exception is roll which only has NOT_TRIGGERED[0]
        # TODO: this is wrong. wrist joints flipped
        joint_limit_switch_dims = np.concatenate((np.full(num_joints - 1, 3), np.array([1])))

        # TODO: Load mechanical limits from something (ex. pull info from config in uwrt_mars_rover thru git)
        self.observation_space = spaces.Dict({
            'goal': spaces.Dict({
                'desired_key_pose_in_world_frame': spaces.Dict({
                    'position': spaces.Box(low=np.full(3, -np.inf), high=np.full(3, np.inf), shape=(3,),
                                           dtype=np.float32),
                    'orientation': spaces.Box(low=np.full(4, -np.inf), high=np.full(4, np.inf), shape=(4,),
                                              dtype=np.float32),
                }),
                'initial_distance_to_target': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
                'initial_orientation_difference': spaces.Box(low=np.full(4, -np.inf), high=np.full(4, np.inf),
                                                             shape=(4,), dtype=np.float32)
            }),
            'joint_sensors': spaces.Dict({
                # Order of array is [turntable, shoulder, elbow, wrist pitch, wrist roll]
                # TODO: this is wrong. wrist joints flipped
                'position': spaces.Box(low=np.full(num_joints, -180), high=np.full(num_joints, 180),
                                       shape=(num_joints,), dtype=np.float32),
                'velocity': spaces.Box(low=np.full(num_joints, -np.inf), high=np.full(num_joints, np.inf),
                                       shape=(num_joints,), dtype=np.float32),
                'effort': spaces.Box(low=np.full(num_joints, -np.inf), high=np.full(num_joints, np.inf),
                                     shape=(num_joints,), dtype=np.float32),
                'joint_limit_switches': spaces.MultiDiscrete(joint_limit_switch_dims),
                'joint_vel_limit_switches': spaces.MultiDiscrete(joint_limit_switch_dims),
            }),
        })

        self.action_space = spaces.Dict({
            'joint_commands': spaces.Box(low=np.full(num_joints, -3), high=np.full(num_joints, 1),
                                                  shape=(num_joints,), dtype=np.float32)
        })

        self.observation = {
            'goal': {
                'desired_key_pose_in_world_frame': {
                    'position': self.init_options.key_position,
                    'orientation': self.init_options.key_orientation,
                },
                'initial_distance_to_target': np.array(np.inf),
                'initial_orientation_difference': np.full(4, np.inf),
            },
            'joint_sensors': {
                'position': np.zeros(num_joints),
                'velocity': np.zeros(num_joints),
                'effort': np.zeros(num_joints),
                'joint_limit_switches': np.zeros(num_joints),
                'joint_vel_limit_switches': np.zeros(num_joints),
            }
        }

        self.info = {
            'sim': {
                'max_steps': self.init_options.max_steps,
                'steps_executed': 0,
                'seconds_executed': 0,
                'end_condition': 'Not Done',
                'keys_hit': 0,
            },
            'goal': {
                'distance_to_target': 0,
                'previous_distance_to_target': 0,
                'distance_moved_towards_target': 0,
                'orientation_difference': [0, 0, 0, 0],
            },
            'arm': {
                'allen_key_pose_in_world_frame': {
                    'position': [0, 0, 0],
                    'orientation': [0, 0, 0, 0],
                },
                'num_joints': num_joints,
                'actuated_joints_names': actuated_joints_names,
                'num_actuated_joints': num_actuated_joints,
                'joint_limits': joint_limits,
                'joint_vel_limits': joint_vel_limits,
            },
        }

    def __spawn_key(self):
        """ Randomize keyboard """
        self.keyboard_orientation = np.array([0, 0, 0, 1])

        if self.init_options.is_keyboard_demo_env:
            """ Select a subset of key locations that we know the arm can hit """
            TEST_KEY_POSITIONS = np.array([[0.85, -0.1, 0.775],  # O
                                           [0.85, -0.0, 0.725],  # N
                                           [0.85, +0.1, 0.775],  # T
                                           [0.85, -0.1, 0.775],  # O
                                           [0.85, -0.05, 0.725],  # M
                                           [0.85, +0.175, 0.750],  # A
                                           [0.85, +0.125, 0.775],  # R
                                           [0.85, +0.155, 0.750],  # S
                                           [0.85, +0.185, 0.775],  # !
                                           [0.85, +0.185, 0.800],  # On To Mars!
                                           ])

            TEST_KEY_TEXT = np.array(['O',
                                      'N',
                                      'T',
                                      'O',
                                      'M',
                                      'A',
                                      'R',
                                      'S',
                                      '!',
                                      'On To Mars!'])
            self.keyboard_position = np.array(TEST_KEY_POSITIONS[self.info['sim']['keys_hit']]).flatten()
            self.keyboard_text = np.str(TEST_KEY_TEXT[self.info['sim']['keys_hit']])
            self.info['sim']['keys_hit'] += 1

        elif self.init_options.is_val_env:
            """ Select a subset of key locations that we know the arm can hit """
            VAL_KEY_POSITIONS = np.array([[0.85, 0, 0.8],
                                          [0.85, 0.2, 0.7],
                                          [0.85, -0.2, 0.7],
                                          [0.9, 0, 0.7]])
            random_idx = np.random.randint(low=0, high=4, size=1)
            self.keyboard_position = np.array(VAL_KEY_POSITIONS[random_idx]).flatten()

        else:
            """ Randomize keyboard based on workspace analysis """
            self.keyboard_position = np.array([np.random.uniform(0.8, 0.9),
                                               np.random.uniform(-0.30, 0.30),
                                               np.random.uniform(0.65, 0.85)])

        self.observation = {
            'goal': {
                'desired_key_pose_in_world_frame': {
                    'position': self.keyboard_position,
                    'orientation': self.keyboard_orientation,
                }
            }
        }

    def __gazebo_observation_to_rs_state(self):
        rs_state = []
        rs_state.extend(list(self.observation['goal']['desired_key_pose_in_world_frame']['position']))
        rs_state.extend(list(self.observation['goal']['desired_key_pose_in_world_frame']['orientation']))
        rs_state.extend(list(self.observation['joint_sensors']['position']))
        rs_state.extend(list(self.observation['joint_sensors']['velocity']))
        rs_state.extend(list(self.observation['joint_sensors']['effort']))
        rs_state.extend(list(self.info['arm']['allen_key_pose_in_world_frame']['position']))
        rs_state.extend(list(self.info['arm']['allen_key_pose_in_world_frame']['orientation']))
        return rs_state

    def __update_observation_and_info(self, rs_state, reset=False):

        arm_joint_positions = rs_state[7:12]
        arm_joint_velocities = rs_state[12:17]
        arm_joint_torques = rs_state[17:22]

        arm_joint_limit_states = [
            1 if arm_joint_positions[joint_index] <= self.info['arm']['joint_limits'][joint_index][0] else
            2 if arm_joint_positions[joint_index] >= self.info['arm']['joint_limits'][joint_index][1] else
            0 for joint_index in range(self.info['arm']['num_joints'])]
        arm_joint_vel_limit_states = [
            1 if arm_joint_velocities[joint_index] <= self.info['arm']['joint_vel_limits'][joint_index][0] else
            2 if arm_joint_velocities[joint_index] >= self.info['arm']['joint_vel_limits'][joint_index][1] else
            0 for joint_index in range(self.info['arm']['num_joints'])]

        self.observation['joint_sensors'] = {
            "position": arm_joint_positions,
            "velocity": arm_joint_velocities,
            "effort": arm_joint_torques,
            "joint_limit_switches": arm_joint_limit_states,
            'joint_vel_limit_switches': arm_joint_vel_limit_states,
        }

        allen_key_tip_position_world_frame = rs_state[22:25]
        allen_key_tip_orientation_world_frame = rs_state[25:29]
        self.info['arm']['allen_key_pose_in_world_frame'] = {
            'position': allen_key_tip_position_world_frame,
            'orientation': allen_key_tip_orientation_world_frame,
        }

        distance_to_target = np.array(np.linalg.norm(
            allen_key_tip_position_world_frame - \
            self.observation['goal']['desired_key_pose_in_world_frame']['position']),
            dtype=np.float32)

        self.info['goal']['previous_distance_to_target'] = self.info['goal']['distance_to_target']
        self.info['goal']['distance_to_target'] = distance_to_target
        self.info['goal']['distance_moved_towards_target'] = self.info['goal']['previous_distance_to_target'] - \
                                                             self.info['goal']['distance_to_target']

        # Difference in Quaternion
        # difference_quaternion = np.array(pb.getDifferenceQuaternion(allen_key_tip_orientation_world_frame,
        #                                                             self.observation['goal']['desired_key_pose_in_world_frame']
        #                                                             ['orientation']), dtype=np.float32)

        # Difference in Rotation Matrix
        current_rotation_matrix = R.from_quat(allen_key_tip_orientation_world_frame).as_matrix()
        goal_rotation_matrix = R.from_quat(self.observation['goal']['desired_key_pose_in_world_frame']
                                           ['orientation']).as_matrix()

        # Now R*R' should produce eye(3)
        rotation_vector = R.from_matrix(current_rotation_matrix.dot(goal_rotation_matrix.T)).as_rotvec()
        rotation_error = np.pi - np.linalg.norm(rotation_vector)  # in rads
        percentage_rotation_error = rotation_error / np.pi  # normalized from 0 to 1 as a %

        # self.info['goal']['orientation_difference'] = difference_quaternion
        self.info['goal']['percentage_rotation_error'] = percentage_rotation_error

        if reset:
            self.info['sim']['steps_executed'] = 0

            self.observation['goal']['initial_distance_to_target'] = self.info['goal']['distance_to_target']
            self.observation['goal']['initial_orientation_difference'] = self.info['goal']['orientation_difference']
        else:
            self.info['sim']['steps_executed'] += 1
            # print(f"steps_executed: {self.info['sim']['steps_executed']}")

    def __clip_action(self, action):
        # from network
        action = action['joint_commands'] if isinstance(action, dict) else action

        ####################
        # pos control
        ####################

        clipped_action = []
        for joint_index in range(self.info['arm']['num_joints']):
            clipped_action.append(np.clip(action[joint_index],
                                          self.info['arm']['joint_limits'][joint_index][0],
                                          self.info['arm']['joint_limits'][joint_index][1]))

        return np.array(clipped_action)

    def __calculate_reward(self):
        percent_time_used = self.info['sim']['steps_executed'] / self.info['sim']['max_steps']
        percent_distance_remaining = self.info['goal']['distance_to_target'] / \
                                     self.observation['goal']['initial_distance_to_target']

        # TODO: scale based off max speed to normalize
        distance_moved = self.info['goal']['distance_moved_towards_target'] / self.observation['goal']['initial_distance_to_target']

        distance_weight = 1
        time_weight = 1 - distance_weight

        # TODO: investigate weird values
        # reward = distance_moved * UWRTArmEnv.REWARD_MAX / 2
        reward = (1 - percent_distance_remaining) * UWRTArmEnv.REWARD_MAX / 2

        # TODO (ak): tweak reward formula to reward more for orientation thats closer to perpendicular to surface of key
        percentage_rotation_error = self.info['goal']['percentage_rotation_error']
        reward -= percentage_rotation_error * UWRTArmEnv.REWARD_MAX / 10

        if self.info['goal']['distance_to_target'] < UWRTArmEnv.GOAL_POSITION_DISTANCE_THRESHOLD:
            self.info['sim']['end_condition'] = 'Key Reached'
            done = True
            reward += UWRTArmEnv.REWARD_MAX / 2

        elif self.info['sim']['steps_executed'] >= self.info['sim']['max_steps']:
            self.info['sim']['end_condition'] = 'Max Sim Steps Executed'
            done = True
            reward -= UWRTArmEnv.REWARD_MAX / 2
        else:
            done = False

        # TODO: add penalty for hitting anything that's not the desired key

        return reward, done

    def reset(self, initial_joint_positions=None, ee_target_pose=None):

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))
        # print("rs_state: {},\n{}".format(len(rs_state), np.array(rs_state).reshape(-1)))

        self.__spawn_key()
        self.__update_observation_and_info(rs_state, reset=True)
        rs_state = self.__gazebo_observation_to_rs_state()
        # print("uwrt_arm_home_pose: {},\n{}".format(len(uwrt_arm_home_pose), np.array(uwrt_arm_home_pose).reshape(-1)))

        # Set initial state of the Robot Server
        state_msg = robot_server_pb2.State(state=rs_state)
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))
        # print("rs_state: {},\n{}".format(len(rs_state), np.array(rs_state).reshape(-1)))

        return self.observation

    def step(self, action):

        # Convert environment action to Robot Server action
        # TODO: scale action with joint_vel_limits
        rs_action = self.__clip_action(copy.deepcopy(action))

        # Send action to Robot Server
        if not self.client.send_action(rs_action.tolist()):
            raise RobotServerError("send_action")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))
        self.__update_observation_and_info(rs_state)

        reward, done = self.__calculate_reward()

        return self.observation, reward, done, self.info

    def render(self):
        pass

class UWRTArmSim(UWRTArmEnv, Simulation):
    cmd = "roslaunch uwrt_robot_server uwrt_sim_robot_server.launch"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        UWRTArmEnv.__init__(self, rs_address=self.robot_server_ip, **kwargs)