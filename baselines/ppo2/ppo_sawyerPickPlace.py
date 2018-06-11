#!/usr/bin/env python3
import numpy as np
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
from railrl.envs.mujoco.sawyer_PickPlace import SawyerPickPlace_Fixed
import time
import pickle
import tensorflow as tf


defGoal = None


def train(env_id, num_timesteps, seed, load_path=None, itr = 1, goal = defGoal, simulate = False, c1 = 10, c2 = 1):
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

    config.gpu_options.allow_growth=True
    
    with tf.Session(config=config) as sess:

        def make_env():
            
            env = gym.make(env_id)

            env.env.setCValues(c1, c2)
            env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)



            return env

        env = DummyVecEnv([make_env])

        loadModel , loadEnv = None, None

        if load_path!=None:
            loadModel = load_path+"/model"+str(itr)
            loadEnv = load_path+"/scaling"+str(itr)+".pkl"

       
        env = VecNormalize(env, loadFile = loadEnv)
        

        set_global_seeds(seed)
        policy = MlpPolicy



        model = ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
                           lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                           ent_coef=0.0,
                           lr=3e-4,
                           cliprange=0.2,
                           total_timesteps=num_timesteps, loadModel=loadModel)

        if simulate:

            obs = np.zeros((env.num_envs,) + env.observation_space.shape)
            obs[:] = env.reset()
            while True:
               
                actions = model.detStep(obs)[0]
                
                obs[:]  = env.step(actions)[0]
                env.render()
                time.sleep(0.05)


    
def simulate(checkpointsDir, itr = 1, goalIndex = None):

    train(env_id, num_timesteps =  0, seed = 1, load_path=dirPREFIX+checkpointsDir, itr = itr,  simulate = True)
                
  

def genGoals():
    goals = [] 
    for i in range(30):
        x = np.random.uniform(-.25, .25)
        y = np.random.uniform(.5, .9)
        goals.append([x,y])

    fobj = open("goals.pkl", "wb")
    pickle.dump(goals, fobj)
    fobj.close()

def readGoals(_file):
    fobj = open(_file, "rb")
    goals = pickle.load(fobj)
    return goals




dirPREFIX = "/home/russellm/baselines/data/"
env_id = 'SawyerPickPlace-v0' ; N = 5* (1e5) ; seed = 1 ; simItr = 50; simGoal = 0
expName = 'SawyerPickPlace-PPO-Batch2048-pathLength150-expoPlacingRew-Tstep.01-fskip5'

for c1 in [0.1, 0.5]:
    for c2 in [1, 0.1, 0.01]:
        tf.reset_default_graph()

        logger.configure(dir = dirPREFIX+expName+"/"+ "c1_"+str(c1)+"_c2_"+str(c2))
        train(env_id, num_timesteps=N, seed = seed, c1 = c1, c2 = c2)

# for goal in goals:
    
#     tf.reset_default_graph()

#     logger.configure(dir = dirPREFIX+expName+"/"+str(count))
#     train(env_id, num_timesteps=N, seed = seed, goal=goal)

   
#     count+=1

#count = 0
#logger.configure(dir = dirPREFIX+expName+"/"+str(count))
#train(env_id, num_timesteps=N, seed = seed)


#simulate(expName+'/'+str(simGoal)+'/checkpoints/', itr = simItr, goalIndex = simGoal)



