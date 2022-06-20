from qdpy import algorithms, containers, plots
from qdpy.base import ParallelismManager
from qdpy.plots import *
import math
import matplotlib.pyplot as plt
import pickle
from os import path
import copy
from absl import app
from absl import flags
from ai_safety_gridworlds.environments.shared import safety_game
from ai_safety_gridworlds.environments.shared import safety_ui
from ai_safety_gridworlds.demonstrations import demonstrations
from ai_safety_gridworlds.helpers import factory
import random
import numpy as np
from pycolab import rendering
import time
from simulate import simulate, Model
import yaml
from qdpy.algorithms import *
from qdpy.containers import *
from qdpy.plots import *
from qdpy.base import *
from qdpy import tools
import sys
import datetime
import pathlib
import shutil
import seaborn as sns
import pandas as pd
import warnings
from main_tw import neat_main
import os
warnings.filterwarnings("ignore")

neat = False


class Model:
    ''' Simple MLP '''
    def __init__(self, config):
        self.layers = config['layers']
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.activation = config['activation_function']
        self.last_action = 0
        self.init_nn()

    def init_nn(self):
        layers_size = [self.input_size] + self.layers + [self.output_size]
        self.shapes = []
        for i in range(1, len(layers_size)):
            fst = layers_size[i-1]
            snd = layers_size[i]
            self.shapes.append((fst, snd))

        self.weight = []
        self.bias = []
        self.param_count = 0

        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            self.param_count += (np.product(shape) + shape[1])

    def get_action(self, x):
        h = np.array(x).flatten()
        nb_layers = len(self.weight)
        for i in range(nb_layers):
            w = self.weight[i]
            b = self.bias[i]
            h = np.matmul(h, w) + b
            h = np.tanh(h)
        # print("h:", h)
        a = np.argmax(h)
        # print(a)
        if a != self.last_action:
            self.last_action = a
            return a
        else:
            h = np.delete(h, np.argmax(h))
            self.last_action = np.argmax(h)
            return np.argmax(h)

    def set_model_params(self, model_params):
        pointer = 0
        for i in range(len(self.shapes)):
            w_shape = self.shapes[i]
            b_shape = self.shapes[i][1]
            s_w = np.product(w_shape)
            s = s_w + b_shape
            chunk = np.array(model_params[pointer:pointer+s])
            self.weight[i] = chunk[:s_w].reshape(w_shape)
            self.bias[i] = chunk[s_w:].reshape(b_shape)
            pointer += s

def eval_fn(ind,config):

    model = Model(config)
    model.set_model_params(ind)
    env = factory.get_environment_obj(config['env_name'])

    scores = simulate(model, env, num_episodes = config['episodes_per_eval'], bc1 = config['feature1'], bc2 =config['feature2'], action_features = config['action_features'],  conf = config)


    if not config['action_features']:
        
        if not config['feature2']:
            
            return (scores["meanAvgReward"],), (scores[config['feature1']] ,0), scores['valid'], scores['ss']

        elif config['feature2'] == "pos_of_rocks":
            #print(scores['pos_of_rocks'])

            return (scores["meanAvgReward"],), (scores[config['feature1']], scores['pos_of_rocks'][0], scores['pos_of_rocks'][1],  scores['pos_of_rocks'][2]), scores['valid'], scores['ss']

        else:
            return (scores["meanAvgReward"],), (scores[config['feature1']], scores[config['feature2']]), scores['valid'], scores['ss']
        
    
    else:
        #if config['algorithm'] == "ME":
             #return (scores["meanAvgReward"],), (scores['actions_ME']), scores['valid'], scores['ss']
            
        #print(scores['actions'])
        return (scores["meanAvgReward"],), (scores['actions']), scores['valid'], scores['ss']




def loadConfig(config_filename):
    config_name = os.path.splitext(os.path.basename(config_filename))[0]
    config = yaml.safe_load(open(config_filename))

    return config


if __name__ == "__main__":

    config_name = sys.argv[1]
    base_path = sys.argv[2] 
    config = loadConfig(config_name)
    date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    experiment_path = base_path + date
    pathlib.Path(experiment_path).mkdir(parents=True, exist_ok=True)
    logs = {}
    n_runs = config['n_runs']
    
    for i in range(n_runs):

        run_path = experiment_path + "/run_" + str(i) + "/"
        pathlib.Path(run_path).mkdir(parents=True, exist_ok=True)
    
        if config['algorithm'] == "ME":
            algo_name = "MAP-Elites"
         
            grid = containers.Grid(shape=config['gridshape'], max_items_per_bin=1, fitness_domain = config['fitness_domain'], features_domain= config['features_domain'])
            
            
        elif config['algorithm'] == "NSLC":
            
            algo_name = "NSLC"
            if config['action_features']:
                 grid = NoveltyArchive( k = config['k'], threshold_novelty= config['threshold_novelty'], fitness_domain=config['fitness_domain'], features_domain = [(0, 4),] * 100, storage_type=list, depot_type=OrderedSet, capacity =  config['capacity'])

            else:
                grid = NoveltyArchive( k = config['k'], threshold_novelty= config['threshold_novelty'], fitness_domain=config['fitness_domain'], features_domain=config['features_domain'], storage_type=list, depot_type=OrderedSet, capacity =  config['capacity'])

        elif config['algorithm'] == "NS":
            algo_name = "NS"
            if config['action_features']:
                grid = NoveltyArchive( k = config['k'], threshold_novelty= config['threshold_novelty'], fitness_domain=config['fitness_domain'], features_domain = [(0, 4),] * 100, storage_type=list, depot_type=OrderedSet, capacity =  config['capacity'], NS = True)
            else:
                
                grid = NoveltyArchive( threshold_novelty= config['threshold_novelty'], fitness_domain= config['fitness_domain'], features_domain=config['features_domain'], storage_type=list, depot_type=OrderedSet, NS = True, capacity = config['capacity'], k = config['k'])

        elif config['algorithm'] == "NEAT":

            neat = True
            logger = neat_main(config)
            logs['run_' + str(i)] = logger
            continue

        model = Model(config)
        dims = model.param_count
        algo_name += "_run_"+ str(i)
    
        algo = algorithms.RandomSearchMutPolyBounded(grid, budget= config['budget'], batch_size=config['batch_size'],
                                                         dimension= dims,  optimisation_task="max", name = algo_name, ind_domain = config['ind_domain'])

      

        # Create a logger to pretty-print everything and generate output data files
        logger = algorithms.TQDMAlgorithmLogger(algo,log_base_path=run_path, save_period = config['save_period'])

        # Run illumination process !
        with ParallelismManager("concurrent") as pMgr: #max_workers = 16
            best = algo.optimise(eval_fn, executor = pMgr.executor, batch_mode= False, config = config)

        #plots.default_plots_grid(logger, output_dir= run_path, conf=config)
        

    #save the logs if needed for further evals

    if not config['algorithm'] == "NEAT":
        logs = {}
        for i in range(n_runs):
            with open(experiment_path + "/run_" + str(i)+ "/final.p", 'rb') as f:
                logs["run_"+str(i)] = pickle.load(f)

    #print(logs)
    #print(experiment_path)

    pickle_out = open(experiment_path + "/logs.pkl", 'wb')
    #print(pickle_out)
    
    pickle.dump(logs,pickle_out)
    pickle_out.close()
    shutil.copy2(config_name, experiment_path)
    
