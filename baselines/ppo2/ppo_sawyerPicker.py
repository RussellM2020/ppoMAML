#!/usr/bin/env python3
import numpy as np
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
from railrl.envs.mujoco.sawyer_gripper import SawyerPick_ObjFixed
import time


def train(env_id, num_timesteps, seed, load_path=None, version = 1):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)

        return env

    env = DummyVecEnv([make_env])

    loadModel , loadEnv = None, None

    if load_path!=None:
        loadModel = load_path+"/model"+str(version)
        loadEnv = load_path+"/scaling"+str(version)+".pkl"

   
    env = VecNormalize(env, loadFile = loadEnv)
    

    set_global_seeds(seed)
    policy = MlpPolicy



    model = ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
                       lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                       ent_coef=0.0,
                       lr=3e-4,
                       cliprange=0.2,
                       total_timesteps=num_timesteps, loadModel=loadModel)

    return model, env

def simulate(checkpointsDir, version = 1):
    model, env = train(env_id, 0, 1, load_path=dirPREFIX+checkpointsDir, version = simVersion)
    #logger.log("Running trained model")
    obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    obs[:] = env.reset()
    while True:
       
        actions = model.detStep(obs)[0]
        obs[:]  = env.step(actions)[0]
        env.render()
        time.sleep(0.002)



dirPREFIX = "/home/russellm/baselines/data/"
env_id = 'SawyerPicker-v0' ; N = 1e6 ; seed = 1 ; simVersion = 80
expName = 'SawyerPicker-PPO-Batch2048'

#logger.configure(dir = dirPREFIX+expName)
#train(env_id, num_timesteps=N, seed = seed)

simulate(expName+'/checkpoints/', version = simVersion)



