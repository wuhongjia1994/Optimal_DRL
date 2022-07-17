import os
import os.path as osp
import time
from collections import deque
from common import utils

import numpy as np
import yaml
from common import logger2 as logger
import config
import tensorflow as tf

from Runner import OnpolicyRunner
from algorithms.mappo import MappoModel
from test_envs import make_top_parallel_env
from types import SimpleNamespace as SN
import random

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#----------------------save_csv_data-------------------


def get_alg_config(name):
    with open(os.path.join(os.path.dirname(config.__file__), name + ".yaml"), "r") as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return SN(**config_dict)


if __name__ == "__main__":

    args = get_alg_config('mappo')
    utils.set_global_seed(args.seed)

    # config = tf.ConfigProto(allow_soft_placement=True)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # config.gpu_options.allow_growth = True
    tf.compat.v1.InteractiveSession()

    env = make_top_parallel_env(args)

    save_dir = 'data/' + args.map_id + utils.get_token(unique=True)
    os.makedirs(save_dir)
    logger.configure(save_dir)

    with open(save_dir + '/args_dict.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)

    nbatch = args.runner_steps
    nbatch_train = nbatch // args.nminibatch
    model = MappoModel(env, args)

    load_dir = None
    model.load(load_dir)

    runner = OnpolicyRunner(env, model, args)

    nupdates = args.n_total_steps // (args.runner_steps * args.nenv_run)
    epinfobuf = deque(maxlen=10)
    start_time = time.time()

    for update in range(1, nupdates + 1):
        tstart = time.time()

        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = utils.decay(args.lr, frac, args.lr_decay)
        cliprangenow = utils.decay(args.cliprange, frac, args.clip_decay)

        # sampling phase
        data, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        # training phase
        mblossvals = []
        for epoch in range(args.noptepochs):
            data.shuffle(args.rnn_chunk)
            for start in range(0, nbatch, nbatch_train):
                mini_data = data[start: start + nbatch_train]
                mblossvals.append(model.train(lrnow, cliprangenow, mini_data))

        # diagnosis
        tnow = time.time()
        ndata = args.runner_steps * args.nenv_run
        fps = int(ndata / (tnow - tstart))
        lossnames = ['pg_loss', 'vf_loss', 'entropy']
        if update % args.log_interval == 0 or update == 1:
            logger.logkv("agent", np.array([agent for agent in range(env.nagent)]))
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", int(update * ndata))
            logger.logkv("fps", fps)
            logger.logkv('average_reward', np.mean([epinfo['r'] for epinfo in epinfobuf], 0).round(2)/128)
            logger.logkv('eplenmean', utils.safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - start_time)
            lossvals = np.mean(mblossvals, axis=0)
            for (lossval, lossname) in zip(lossvals, lossnames):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()

            """这里是新加的"""
            VOI = np.mean([epinfo['VOI'] for epinfo in epinfobuf],0)
            actions = np.mean([epinfo['actions'] for epinfo in epinfobuf],0)
            user_reward = np.mean([epinfo['user_reward'] for epinfo in epinfobuf],0)
            for i in range(env.nagent):
                print('VOI_agent%d'%i, VOI[i])
                print('actions_agent%d'%i, actions[i])
            print('user_reward', user_reward)

            #Data = {'actions': actions[0]}
            # Save_to_Csv(data=Data, file_name='price_MC1', Save_format='csv', Save_type='row')
            #
            # Data_MC2 = {'actions': actions[1]}
            # Save_to_Csv(data=Data_MC2, file_name='price_MC2', Save_format='csv', Save_type='row')
            # Data_MC3 = {'actions': actions[2]}
            # Save_to_Csv(data=Data_MC3, file_name='price_MC3', Save_format='csv', Save_type='row')

        if args.save_interval and (update % args.save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
