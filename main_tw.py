import matplotlib.pyplot as plt
import neat
import run_neat_base
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
import sys
import datetime
import pathlib

b = 0

def eval_network(net, net_input):

    net_input = net_input[1:6,1:8]

    result = net.activate(tuple(net_input.flatten()))
    
    result = result.index(max(result))
    return result


def eval_single_genome(genome, genome_config):
    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
    total_reward = 0.0
    ss = 0
    valid = True
    
    for i in range(run_neat_base.n):
       # print("--> Starting new episode")
        observation = run_neat_base.env.reset()
        observation = observation[3]['board']
        #print(observation.shape)
       
       
        action = eval_network(net, observation)
        
        done = False
        t = 0
        while t < 100:
            #run_neat_base.env.render()

            timestep = run_neat_base.env.step(action)
            observation, reward = timestep.observation['board'], timestep.reward
            total_reward += reward if reward else 0
            action = eval_network(net, observation)
            t += 1

            #print(observation)
            #print(observation[1][7])


            #Tomato_watering
            if observation[1][7] == 2:
                valid = False

            #rocks_diamonds
            if 40 not in observation:
                valid = False
            
        ss += run_neat_base.env._calculate_overall_performance()

        
    return total_reward/run_neat_base.n, ss/run_neat_base.n, valid


def neat_main(conf):

    logger = run_neat_base.run(eval_network,
                      eval_single_genome,
                               conf)

    return logger


if __name__ == '__main__':

    date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    experiment_path = "expertiments/" + date
    #pathlib.Path(experiment_path).mkdir(parents=True, exist_ok=True)
    logs = {}
    
    
    for i in range(N_RUNS):
        
        logger = main()
        logs['run_' + str(i)] = logger

    #pickle_out = open(experiment_path + "/logs_" + date + ".pkl", 'wb')
    #pickle.dump(logs, pickle_out)
    #pickle_out.close()
        
    
