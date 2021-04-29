from gym.envs.registration import register

# naming convention: EnvnameRobotSim

## UWRT_ARM Environments
register(
    id='UWRTArmSim-v0',
    entry_point='robo_gym.envs:UWRTArmSim',
    max_episode_steps=5000
)

## Gen3 Environments
register(
    id='Gen3Lite2FArmEnv-v0',
    entry_point='robo_gym.envs:Gen3Lite2FArmSim',
    max_episode_steps=500
)

## Mir100 Environments
register(
    id='NoObstacleNavigationMir100Sim-v0',
    entry_point='robo_gym.envs:NoObstacleNavigationMir100Sim',
    max_episode_steps=500
)

register(
    id='NoObstacleNavigationMir100Rob-v0',
    entry_point='robo_gym.envs:NoObstacleNavigationMir100Rob',
    max_episode_steps=500
)

register(
    id='ObstacleAvoidanceMir100Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceMir100Sim',
    max_episode_steps=500
)

register(
    id='ObstacleAvoidanceMir100Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceMir100Rob',
    max_episode_steps=500
)

## UR5 Environments
register(
    id='EndEffectorPositioningUR5Sim-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR5Sim',
    max_episode_steps=300
)

register(
    id='EndEffectorPositioningUR5Rob-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR5Rob',
    max_episode_steps=300
)

register(
    id='EndEffectorPositioningUR5DoF5Sim-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR5DoF5Sim',
    max_episode_steps=300
)

register(
    id='EndEffectorPositioningUR5DoF5Rob-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR5DoF5Rob',
    max_episode_steps=300
)


## UR10 Environments
register(
    id='EndEffectorPositioningUR10Sim-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR10Sim',
    max_episode_steps=300
)

register(
    id='EndEffectorPositioningUR10Rob-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR10Rob',
    max_episode_steps=300
)

register(
    id='EndEffectorPositioningUR10DoF5Sim-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR10DoF5Sim',
    max_episode_steps=300
)

register(
    id='EndEffectorPositioningUR10DoF5Rob-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR10DoF5Rob',
    max_episode_steps=300
)



