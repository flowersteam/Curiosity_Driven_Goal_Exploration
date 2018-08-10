import argparse
import os
import logging
import datetime
import json

import numpy as np

from latentgoalexplo.actors import exploactors
from latentgoalexplo.environments import armballs


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s[%(module)s:%(funcName)s:%(lineno)d]  %(message)s")
os.environ["JOBLIB_TEMP_FOLDER"] = "."


def RandomParameterExplorationExperiment(static_env, env_config, n_exploration_iterations, seed, logdir, logger=None):
    logger.info("Starting random parameter exploration")
    a = exploactors.RandomParameterizationExploration(static_env=static_env, **env_config)
    a.reset()
    a.act(n_iter=n_exploration_iterations, render=False)

    logger.info("Exploration over, saving data")
    # We save the set of explored states
    explored_states = np.array(a.outcomes_states)
    np.save(os.path.join(logdir, 'explored_states'), explored_states.astype(np.float16))

    # We terminate the agent.
    a.terminate()


def run_experiments(params):
    logger = logging.getLogger(params['name'])
    logger.info("Instantiating the Environment")


    if params['environment'] == "armballs":
        static_env = armballs.MyArmBalls
        env_config = dict()
        env_config.update({'object_size': params['object_size'], 'distract_size': params['distract_size'],
                           'distract_noise': params['distract_noise'],
                           'n_rbf': params['n_rbf'], 'sdev': params['sdev'], 'n_timesteps': 50})
    else:
        raise NotImplementedError

    if params['test']:
        params['n_exploration_iterations'] = int(5e2)

    # Save experiment arguments
    with open(os.path.join(params['path'], 'config.json'), 'w') as f:
        json.dump(params, f, separators=(',\n', ': '))

    logger.info("Instantiating the Explorator")
    RandomParameterExplorationExperiment(static_env=static_env, env_config=env_config,
                                         n_exploration_iterations=params['n_exploration_iterations'],
                                         seed=params['seed'], logdir=params['path'], logger=logger)


def main():
    parser = argparse.ArgumentParser(prog='Random Parameter Exploration',
                                     description='This script performs a random parameter exploration')

    parser.add_argument('environment', help="the Environment you want to use", type=str,
                        choices=["armball", "armballs", "armstickball", "armstickballs"])

    parser.add_argument('--n_rbf', help="Number of RBF to use", type=int, default=7)
    parser.add_argument('--sdev', help="Standard deviation of RBF", type=float, default=10.)
    parser.add_argument('--object_size', help="Radius of the ball", type=float, default=0.17)
    parser.add_argument('--distract_size', help="Radius of the distractor", type=float, default=0.15)
    parser.add_argument('--distract_noise', help="Radius of the distractor", type=float, default=0.1)

    parser.add_argument('--n_exploration_iterations', help="Number of exploration iterations", type=int,
                        default=int(1e4))

    parser.add_argument('--seed', help="Number of random motor babbling iterations", type=int, default=0)
    parser.add_argument('--path', help='Path to the results folder', type=str, default='.')
    parser.add_argument('--name', help='Name of the experiment', type=str, default='')
    parser.add_argument('-t', '--test', help='Whether to make a (shorter) test run', action="store_true")

    args = vars(parser.parse_args())

    assert os.path.isdir(args['path']), "You provided a wrong path."

    if args['name'] == '':
        args['name'] = ("RPE %s %s" % (args['environment'], str(datetime.datetime.now()))).title()

    if args['test']:
        args['path'] = 'test'
    args['path'] = os.path.join(args['path'], args['name'])
    logger = logging.getLogger(args['name'], )
    logger.setLevel(logging.INFO)

    os.mkdir(args['path'])
    handler = logging.FileHandler(os.path.join(args['path'], 'logs.txt'))
    handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s[%(module)s:%(funcName)s:%(lineno)d]  %(message)s"))
    logger.addHandler(handler)

    run_experiments(args)


if __name__ == "__main__":
    main()
