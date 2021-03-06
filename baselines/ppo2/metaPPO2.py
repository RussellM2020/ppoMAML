import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.common.runners import AbstractEnvRunner








class Task_Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, task = 0):
        sess = tf.get_default_session()

       
        taskScope = "Task"+str(task) 
   
       
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False, taskScope = taskScope)   #The old policy
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True, taskScope = taskScope) # The policy to which we are going to update

       

        A = train_model.pdtype.sample_placeholder([None], name=taskScope+"/A")
        ADV = tf.placeholder(tf.float32, [None], name=taskScope+"/ADV")
        R = tf.placeholder(tf.float32, [None], name=taskScope+"/R")
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name=taskScope+"/OLDNEGLOGPAC")
        OLDVPRED = tf.placeholder(tf.float32, [None], name=taskScope+"/OLDVPRED")
        LR = tf.placeholder(tf.float32, [], name=taskScope+"/LR")
        CLIPRANGE = tf.placeholder(tf.float32, [], name=taskScope+"/CLIPRANGE")



        with tf.name_scope(taskScope+"/Objective"):
            neglogpac = train_model.pd.neglogp(A)
            entropy = tf.reduce_mean(train_model.pd.entropy())

            vpred = train_model.vf
            vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
            vf_losses1 = tf.square(vpred - R)
            vf_losses2 = tf.square(vpredclipped - R)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
            pg_losses = -ADV * ratio
            pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef



        with tf.name_scope(taskScope+"/PolicyGradient"):
          
            #with tf.variable_scope(taskScope+'/model'):
               

               
            params = tf.trainable_variables(scope=taskScope+'/model')
               
            grads = tf.gradients(loss, params)
            if max_grad_norm is not None:
                grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(grads, params))
            trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            _train = trainer.apply_gradients(grads)




        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

     


        self.A =  A
        self.ADV = ADV
        self.R = R


        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.detStep = act_model.detStep
        self.value = act_model.value
        self.initial_state = act_model.initial_state
       
        #tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101


class MetaCtrl_Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        
        taskScope = "Meta"
       
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False, taskScope = taskScope)   #The old policy
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True, taskScope = taskScope) # The policy to which we are going to update

       

        A = train_model.pdtype.sample_placeholder([None], name=taskScope+"/A")
        ADV = tf.placeholder(tf.float32, [None], name=taskScope+"/ADV")
        R = tf.placeholder(tf.float32, [None], name=taskScope+"/R")
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name=taskScope+"/OLDNEGLOGPAC")
        OLDVPRED = tf.placeholder(tf.float32, [None], name=taskScope+"/OLDVPRED")
        LR = tf.placeholder(tf.float32, [], name=taskScope+"/LR")
        CLIPRANGE = tf.placeholder(tf.float32, [], name=taskScope+"/CLIPRANGE")

        

        self.A =  A
        self.ADV = ADV
        self.R = R


       
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.detStep = act_model.detStep
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        
        #tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101




class InnerUpdate_Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, actModel , trainModel , actScope = "Meta", trainScope = "Task1", task=0):
        sess = tf.get_default_session()

     
       
        act_model  = actModel
        train_model = trainModel

       

        A = train_model.pdtype.sample_placeholder([None], name=actScope+"/A")
        ADV = tf.placeholder(tf.float32, [None], name=actScope+"/ADV")
        R = tf.placeholder(tf.float32, [None], name=actScope+"/R")
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name=actScope+"/OLDNEGLOGPAC")
        OLDVPRED = tf.placeholder(tf.float32, [None], name=actScope+"/OLDVPRED")
        LR = tf.placeholder(tf.float32, [], name=actScope+"/LR")
        CLIPRANGE = tf.placeholder(tf.float32, [], name=actScope+"/CLIPRANGE")



        with tf.name_scope("Meta-task"+str(task)+"/Objective"):
            neglogpac = train_model.pd.neglogp(A)
            entropy = tf.reduce_mean(train_model.pd.entropy())

            vpred = train_model.vf
            vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
            vf_losses1 = tf.square(vpred - R)
            vf_losses2 = tf.square(vpredclipped - R)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
            pg_losses = -ADV * ratio
            pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        with tf.name_scope("Meta-task"+str(task)+"/PolicyGradient"):
            #with tf.variable_scope(trainScope+'/model'):
               
            params = tf.trainable_variables(scope =trainScope+'/model' )
            import ipdb
            ipdb.set_trace()

            grads = tf.gradients(loss, params)
            if max_grad_norm is not None:
                grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(grads, params))
            trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            _train = trainer.apply_gradients(grads)




        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

       

        self.A =  A
        self.ADV = ADV
        self.R = R


        #self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.detStep = act_model.detStep
        self.value = act_model.value
        self.initial_state = act_model.initial_state
      
        #tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101



class OuterUpdate_Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, metaController, numTasks = 2):
        sess = tf.get_default_session()

     
       
       
        act_model = metaController.act_model
        train_model = metaController.train_model
        actScope = "Meta"
        
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name=actScope+"/OLDNEGLOGPAC")
        
        LR = tf.placeholder(tf.float32, [], name=actScope+"/LR")
        CLIPRANGE = tf.placeholder(tf.float32, [], name=actScope+"/CLIPRANGE")

        paramList = []
        for task in range(numTasks):

            dataScope = "Task" + str(task)

            A = train_model.pdtype.sample_placeholder([None], name=dataScope+"/A")
            ADV = tf.placeholder(tf.float32, [None], name=dataScope+"/ADV")
            R = tf.placeholder(tf.float32, [None], name=dataScope+"/R")
            OLDVPRED = tf.placeholder(tf.float32, [None], name=dataScope+"/OLDVPRED")
            paramList.append([A, ADV, R, OLDVPRED])


        with tf.name_scope("OuterUpdate/Objective"):

            overallLoss = 0
            for setting in paramList:
                A, ADV, R, OLDVPRED = setting[0], setting[1], setting[2], setting[3]
             
                neglogpac = train_model.pd.neglogp(A)
                entropy = tf.reduce_mean(train_model.pd.entropy())

                vpred = train_model.vf
                vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
                vf_losses1 = tf.square(vpred - R)
                vf_losses2 = tf.square(vpredclipped - R)
                vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
                pg_losses = -ADV * ratio
                pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
                pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
                clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
                loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
                overallLoss += loss

        with tf.name_scope("OuterUpdate/PolicyGradient"):
            #with tf.variable_scope(trainScope+'/model'):
               
            params = tf.trainable_variables(scope ='Meta/model' )
            import ipdb
            ipdb.set_trace()

            grads = tf.gradients(overallLoss, params)
            if max_grad_norm is not None:
                grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(grads, params))
            trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            _train = trainer.apply_gradients(grads)




        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

     

class Runner(AbstractEnvRunner):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        
     

        for _ in range(self.nsteps):
           
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)


            self.obs[:], rewards, self.dones, infos = self.env.step(actions)


           
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

           

            mb_rewards.append(rewards)



        
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=10, loadModel=None, pickLength = 100):
    
   
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

  
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches



    task_model = lambda task: Task_Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, task = task)

    meta_model = lambda : MetaCtrl_Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)


    innerUpdate = lambda actModel, trainModel, trainScope, task: InnerUpdate_Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,  actModel = actModel, trainModel = trainModel, task=task, trainScope = trainScope)

    outerUpdate = lambda numTasks, metaController: OuterUpdate_Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, numTasks = numTasks, metaController = metaController)


    # make_MetaModel = lambda task: MetaModel(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
    #                 nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
    #                 max_grad_norm=max_grad_norm)
    
    def setupCompGraph():

  

        task1 = task_model(0)
        task2 = task_model(1)

        metaController = meta_model()
        
        innerUpdate(actModel = metaController.act_model, trainModel = task1.train_model, trainScope = "Task0", task = 0)
        innerUpdate(actModel = metaController.act_model, trainModel = task2.train_model, trainScope = "Task1", task = 1)

        outerUpdate(numTasks = 2, metaController = metaController)
       

        sess = tf.get_default_session()
        writer = tf.summary.FileWriter("/home/russellm/tfGraphs/metaPPO")
        writer.add_graph(sess.graph)

    setupCompGraph()
    

    # if save_interval and logger.get_dir():
    #     import cloudpickle
    #     with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
    #         fh.write(cloudpickle.dumps(make_model))

   


   

    # if loadModel is not None:
       

    #     model.load(loadModel)


    #     print("LOADING COMPLETE")
    # runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    # epinfobuf = deque(maxlen=100)
    # tfirststart = time.time()

    

    # nupdates = total_timesteps//nbatch
    # for update in range(1, nupdates+1):
    #     assert nbatch % nminibatches == 0
    #     nbatch_train = nbatch // nminibatches
    #     tstart = time.time()
    #     frac = 1.0 - (update - 1.0) / nupdates
    #     lrnow = lr(frac)
    #     cliprangenow = cliprange(frac)
    #     obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632

        

    #     epinfobuf.extend(epinfos)
    #     mblossvals = []
    #     if states is None: # nonrecurrent version
    #         inds = np.arange(nbatch)
    #         for _ in range(noptepochs):
    #             np.random.shuffle(inds)
    #             for start in range(0, nbatch, nbatch_train):

    #                 end = start + nbatch_train
    #                 mbinds = inds[start:end]
    #                 slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    

                    
    #                 mblossvals.append(model.train(lrnow, cliprangenow, *slices))
    #     else: # recurrent version
    #         assert nenvs % nminibatches == 0
    #         envsperbatch = nenvs // nminibatches
    #         envinds = np.arange(nenvs)
    #         flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
    #         envsperbatch = nbatch_train // nsteps
    #         for _ in range(noptepochs):
    #             np.random.shuffle(envinds)
    #             for start in range(0, nenvs, envsperbatch):
    #                 end = start + envsperbatch
    #                 mbenvinds = envinds[start:end]
    #                 mbflatinds = flatinds[mbenvinds].ravel()
    #                 slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
    #                 mbstates = states[mbenvinds]
    #                 mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

    #     lossvals = np.mean(mblossvals, axis=0)
    #     tnow = time.time()
    #     fps = int(nbatch / (tnow - tstart))
    #     if update % log_interval == 0 or update == 1:
    #         ev = explained_variance(values, returns)
    #         logger.logkv("serial_timesteps", update*nsteps)
    #         logger.logkv("nupdates", update)
    #         logger.logkv("total_timesteps", update*nbatch)
    #         logger.logkv("fps", fps)
    #         logger.logkv("explained_variance", float(ev))
            

    #         logger.logkv('eprewStage1max', safemean([epinfo['maxRewStage1'] for epinfo in epinfobuf]))
    #         logger.logkv('eprewStage2max', safemean([epinfo['maxRewStage2'] for epinfo in epinfobuf]))

    #         logger.logkv('eprewStage1mean', safemean([epinfo['rewStage1'] for epinfo in epinfobuf]))
    #         logger.logkv('eprewStage2mean', safemean([epinfo['rewStage2'] for epinfo in epinfobuf]))
            
          



    #         logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))

            
    #         logger.logkv('time_elapsed', tnow - tfirststart)
    #         for (lossval, lossname) in zip(lossvals, model.loss_names):
    #             logger.logkv(lossname, lossval)
    #         logger.dumpkvs()
    #     if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
    #         checkdir = osp.join(logger.get_dir(), 'checkpoints')
    #         os.makedirs(checkdir, exist_ok=True)
    #         #savepath = osp.join(checkdir, '%.5i'%update)

           
    #         print('Saving env scaling')
    #         env.saveScaling(checkdir+"/scaling"+str(update)+".pkl")

    #         print('Saving model')

    #         model.save(checkdir+"/model"+str(update))
    # env.close()
    # return model

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
