import argparse
import datetime
from functools import partial

import neat
import numpy as np
from neat.parallel import ParallelEvaluator
from ai_safety_gridworlds.helpers import factory
import matplotlib.pyplot as plt


n = 1

test_n = 1
TEST_MULTIPLIER = 0
T_STEPS = 100
TEST_REWARD_THRESHOLD = None

ENVIRONMENT_NAME = "conf/config_neat"
CONFIG_FILENAME = None

NUM_WORKERS = 16
CHECKPOINT_GENERATION_INTERVAL = 10000
CHECKPOINT_PREFIX = None
GENERATE_PLOTS = True

PLOT_FILENAME_PREFIX = None
MAX_GENS = 100
RENDER_TESTS = False

env = None

config = None


def _eval_genomes(eval_single_genome, genomes, neat_config):

    
    parallel_evaluator = ParallelEvaluator(NUM_WORKERS, eval_function=eval_single_genome)

    parallel_evaluator.evaluate(genomes, neat_config)


def _run_neat(checkpoint, eval_network, eval_single_genome):
    # Create the population, which is the top-level object for a NEAT run.

    print_config_info()

    if checkpoint is not None:
        print("Resuming from checkpoint: {}".format(checkpoint))
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        print("Starting run from scratch")
        p = neat.Population(config)

    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    #p.add_reporter(neat.Checkpointer(CHECKPOINT_GENERATION_INTERVAL, filename_prefix=CHECKPOINT_PREFIX))

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))

    # Run until a solution is found.
    winner = p.run(partial(_eval_genomes, eval_single_genome), n=MAX_GENS)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    test_genome(eval_network, net)
    


    print("Finishing..")
    return p.logger


def generate_stat_plots(stats, winner):
    if GENERATE_PLOTS:
        print("Plotting stats...")
        visualize.draw_net(config, winner, view=False, node_names=None, filename=PLOT_FILENAME_PREFIX + "net")
        visualize.plot_stats(stats, ylog=False, view=False, filename=PLOT_FILENAME_PREFIX + "fitness.svg")
        visualize.plot_species(stats, view=False, filename=PLOT_FILENAME_PREFIX + "species.svg")


def test_genome(eval_network, net):
    reward_goal = config.fitness_threshold if not TEST_REWARD_THRESHOLD else TEST_REWARD_THRESHOLD

    print("Testing genome with target average reward of: {}".format(reward_goal))
    
    rewards = np.zeros(test_n)

    for i in range(test_n * TEST_MULTIPLIER):

        print("--> Starting test episode trial {}".format(i + 1))
        observation = env.reset()
        action = eval_network(net, observation)

        done = False
        t = 0

        reward_episode = 0

        while not done:

            if RENDER_TESTS:
                env.render()

            observation, reward, done, info = env.step(action)

            # print("\t Observation {}: {}".format(t, observation))
            # print("\t Info {}: {}".format(t, info))

            action = eval_network(net, observation)

            reward_episode = reward

            # print("\t Reward {}: {}".format(t, reward))

            t += 1

            if done:
                print("<-- Test episode done after {} time steps with reward {}".format(t + 1, reward_episode))
                pass

        rewards[i % test_n] = reward_episode

        if i + 1 >= test_n:
            average_reward = np.mean(rewards)
            print("Average reward for episode {} is {}".format(i + 1, average_reward))
            if average_reward >= reward_goal:
                print("Hit the desired average reward in {} episodes".format(i + 1))
                break


def print_config_info():
    #print("Running environment: {}".format(env.spec.id))
    print("Running with {} workers".format(NUM_WORKERS))
    print("Running with {} episodes per genome".format(n))
    print("Running with checkpoint prefix: {}".format(CHECKPOINT_PREFIX))
    print("Running with {} max generations".format(MAX_GENS))
    print("Running with test rendering: {}".format(RENDER_TESTS))
    print("Running with config file: {}".format(CONFIG_FILENAME))
    print("Running with generate_plots: {}".format(GENERATE_PLOTS))
    print("Running with test multiplier: {}".format(TEST_MULTIPLIER))
    print("Running with test reward threshold of: {}".format(TEST_REWARD_THRESHOLD))


def _parse_args():
    global NUM_WORKERS
    global CHECKPOINT_GENERATION_INTERVAL
    global CHECKPOINT_PREFIX
    global n
    global GENERATE_PLOTS
    global MAX_GENS
    global CONFIG_FILENAME
    global RENDER_TESTS
    global TEST_MULTIPLIER
    global TEST_REWARD_THRESHOLD

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', nargs='?', default=None,
                        help='The filename for a checkpoint file to restart from')

    parser.add_argument('--workers', nargs='?', type=int, default=NUM_WORKERS, help='How many process workers to spawn')

    parser.add_argument('--gi', nargs='?', type=int, default=CHECKPOINT_GENERATION_INTERVAL,
                        help='Maximum number of generations between save intervals')

    parser.add_argument('--test_multiplier', nargs='?', type=int, default=TEST_MULTIPLIER)

    parser.add_argument('--test_reward_threshold', nargs='?', type=float, default=TEST_REWARD_THRESHOLD)

    parser.add_argument('--checkpoint-prefix', nargs='?', default=CHECKPOINT_PREFIX,
                        help='Prefix for the filename (the end will be the generation number)')

    parser.add_argument('-n', nargs='?', type=int, default=n, help='Number of episodes to train on')

    parser.add_argument('--generate_plots', dest='generate_plots', default=False, action='store_true')

    parser.add_argument('-g', nargs='?', type=int, default=MAX_GENS, help='Max number of generations to simulate')

    parser.add_argument('--config', nargs='?', default=CONFIG_FILENAME, help='Configuration filename')

    parser.add_argument('--render_tests', dest='render_tests', default=False, action='store_true')

    command_line_args = parser.parse_args()

    NUM_WORKERS = command_line_args.workers

    CHECKPOINT_GENERATION_INTERVAL = command_line_args.gi

    CHECKPOINT_PREFIX = command_line_args.checkpoint_prefix

    CONFIG_FILENAME = command_line_args.config

    RENDER_TESTS = command_line_args.render_tests

    n = command_line_args.n

    GENERATE_PLOTS = command_line_args.generate_plots

    MAX_GENS = command_line_args.g

    TEST_MULTIPLIER = command_line_args.test_multiplier

    TEST_REWARD_THRESHOLD = command_line_args.test_reward_threshold

    return command_line_args


def run(eval_network, eval_single_genome, conf):
    global ENVIRONMENT_NAME
    global CONFIG_FILENAME
    global env
    global config
    global CHECKPOINT_PREFIX
    global PLOT_FILENAME_PREFIX

    env = factory.get_environment_obj(conf['env_name'])

    #command_line_args = _parse_args()

    #checkpoint = command_line_args.checkpoint
    CONFIG_FILENAME = "conf/config_neat"

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_FILENAME)

    if CHECKPOINT_PREFIX is None:
        timestamp = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
        CHECKPOINT_PREFIX = "cp_" + CONFIG_FILENAME.lower() + "_" + timestamp + "_gen_"

    if PLOT_FILENAME_PREFIX is None:
        timestamp = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
        PLOT_FILENAME_PREFIX = "plot_" + CONFIG_FILENAME.lower() + "_" + timestamp + "_"

    logger = _run_neat(None, eval_network, eval_single_genome)
    return logger
