import tensorflow as tf
from types import SimpleNamespace as SN
import os
import config
import yaml
from test_envs import make_top_parallel_env
from algorithms.mappo import MappoModel
import numpy as np
from collections import deque

from Runner import OnpolicyRunner
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_alg_config(load_dir):
    with open(load_dir + "/args_dict.yaml", "r") as file:
        config_dict = yaml.load(file,Loader = yaml.FullLoader)
    return SN(**config_dict)
if __name__=="__main__":
    tf.InteractiveSession()
    load_dir = 'data/2s3z_2022_5_31_15_19_6'
    args = get_alg_config(load_dir)
    env = make_top_parallel_env(args)

    model = MappoModel(env, args)
    check_points = load_dir + '/checkpoints/00400'
    model.load(check_points)

    runner = OnpolicyRunner(env, model, args)
    nepisode = 10
    epinfobuf = deque(maxlen=nepisode)
    while True:
        print(1)
        epinfos = runner.run()[-1]
        epinfobuf.extend(epinfos)
        if len(epinfobuf) == epinfobuf.maxlen:
            print('average_reward', np.mean([epinfo['r'] for epinfo in epinfobuf], 0).round(2))
            print('mean_episode_length', np.mean([epinfo['l'] for epinfo in epinfobuf]))
            break