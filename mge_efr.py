import argparse
import os
import logging
import datetime
import json
import pickle

import numpy as np

from explauto.utils import prop_choice

from latentgoalexplo.actors import exploactors
from latentgoalexplo.environments import armballs
from latentgoalexplo.environments.explautoenv import ExplautoEnv
from latentgoalexplo.curiosity.learning_module import LearningModule

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s[%(module)s:%(funcName)s:%(lineno)d]  %(message)s")
os.environ["JOBLIB_TEMP_FOLDER"] = "."


def ModularGoalExplorationFIExperiment(static_env, env_config, explauto_config, representation, interest_model,
                                       n_explore, explo_ratio, explo_noise_sdev, win_size, n_exploration_iterations,
                                       n_bootstrap, seed, logdir='test', logger=None):
    np.random.seed(seed)

    logger.info("Bootstrapping phase")
    a = exploactors.RandomParameterizationExploration(static_env=static_env, **env_config)
    a.reset()
    a.act(n_iter=n_bootstrap, render=False)

    # Define motor and sensory spaces:
    explauto_env = ExplautoEnv(**explauto_config)
    m_ndims = explauto_env.conf.m_ndims  # number of motor parameters
    m_space = range(m_ndims)

    # We divide the explo noise by 2 to match explauto implementation with respect to our implementation
    explo_noise_sdev = explo_noise_sdev / 2
    # Create the learning modules:
    learning_modules = []
    if representation == 'flat':
        s_distractball = range(m_ndims, m_ndims + 4)
        learning_modules.append(LearningModule("mod1", m_space, s_distractball, explauto_env.conf,
                                               explo_noise=explo_noise_sdev, win_size=win_size,
                                               interest_model=interest_model))
    elif representation == 'modular':
        s_distract = range(m_ndims, m_ndims + 2)
        s_ball = range(m_ndims + 2, m_ndims + 4)
        learning_modules.append(LearningModule("mod1", m_space, s_distract, explauto_env.conf,
                                               explo_noise=explo_noise_sdev, win_size=win_size,
                                               interest_model=interest_model))
        learning_modules.append(LearningModule("mod2", m_space, s_ball, explauto_env.conf,
                                               explo_noise=explo_noise_sdev, win_size=win_size,
                                               interest_model=interest_model))
    else:
        raise NotImplementedError

    # We update the learning modules with the bootstrap outcomes
    for i, m in enumerate(a.actions):
        s = a.outcomes[i]
        for module in learning_modules:
            module.update_sm(m, module.get_s(np.concatenate([m, s])))

    env = static_env(**env_config)
    env.reset()

    outcomes_states = a.outcomes_states
    interests_evolution = []
    explo_evolution = []
    goals_states = []

    logger.info("Starting exploration")
    # Steps of (4 exploring and 1 exploiting iterations):
    for step in range(n_exploration_iterations // (n_explore + 1)):
        if (step + 1) % 100 == 0:
            logger.info("Iteration: %i / %i" % ((step+1) * (n_explore + 1), n_exploration_iterations))
        # Compute the interest of modules
        interests = [module.interest() for module in learning_modules]
        interests_evolution.append(interests)
        # Choose the babbling module (probabilities proportional to interests, with epsilon of random choice):
        babbling_choice = prop_choice(interests, eps=explo_ratio)
        babbling_module = learning_modules[babbling_choice]
        # The babbling module picks a random goal in its sensory space and returns 4 noisy motor commands:
        m_list = babbling_module.produce(n=n_explore)
        goal = babbling_module.s
        goals_states.append([babbling_choice, goal])
        for m in m_list:
            env.reset()
            env.act(action=m, render=False)
            s = env.observation
            outcomes_states += [env.hidden_state]
            # Update each sensorimotor models:
            for module in learning_modules:
                module.update_sm(m, module.get_s(np.concatenate([m, s])))
        # Choose the best motor command to reach current goal (with no noise):
        m = babbling_module.infer(babbling_module.expl_dims, babbling_module.inf_dims,
                                  babbling_module.x, n=1, explore=False)
        env.reset()
        env.act(action=m, render=False)
        s = env.observation
        outcomes_states += [env.hidden_state]
        # Update the interest of the babbling module:
        babbling_module.update_im(m, babbling_module.get_s(np.concatenate([m, s])))
        # Update each sensorimotor models:
        for module in learning_modules:
            module.update_sm(m, module.get_s(np.concatenate([m, s])))
        explos_modules = [int(100. * (n_explore + 1) * module.im.n_points() / float(module.sm.t)) for module in learning_modules]
        explo_evolution.append(explos_modules)


    logger.info("Exploration finished, saving data")
    # We save the set of explored states and interests evolution for each representation
    explored_states = np.array(outcomes_states)
    np.save(os.path.join(logdir, 'explored_states'), explored_states.astype(np.float32))
    interests_evolution = np.array(interests_evolution)
    np.save(os.path.join(logdir, 'interests_evolution'), interests_evolution.astype(np.float32))
    explo_evolution = np.array(explo_evolution)
    np.save(os.path.join(logdir, 'explo_evolution'), explo_evolution.astype(np.float32))
    # We save the set of goals states
    with open(logdir + '/goal_states', 'wb') as f:
        pickle.dump(goals_states, f)



def run_experiments(params):

    logger = logging.getLogger(params['name'])

    if params['environment'] == "armballs":
        static_env = armballs.MyArmBalls
        env_config = dict()
        env_config.update({'arm_lengths': np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]),
                           'object_size': params['object_size'], 'distract_noise': params['distract_noise'],
                           'n_rbf': params['n_rbf'], 'sdev': params['sdev'], 'n_timesteps': 50})
        explauto_config = dict(
                m_mins=[-1.] * 7 * params['n_rbf'],
                m_maxs=[1.] * 7 * params['n_rbf'],
                s_mins=[-1.] * 4,
                s_maxs=[1.] * 4
        )
    else:
        raise NotImplementedError

    logger.info("Instantiating the Environment")
    if params['test']:
        params['n_exploration_iterations'] = int(1e3)

    with open(os.path.join(params['path'], 'config.json'), 'w') as f:
        json.dump(params, f, separators=(',\n', ': '))

    logger.info("Instantiating the Explorator")
    ModularGoalExplorationFIExperiment(static_env=static_env, env_config=env_config, explauto_config=explauto_config,
                                       representation=params['representation'], interest_model=params['interest_model'],
                                       n_explore=params['n_explore'], explo_ratio=params['explo_ratio'],
                                       explo_noise_sdev=params['explo_noise_sdev'], win_size=params['win_size'],
                                       n_exploration_iterations=params['n_exploration_iterations'],
                                       n_bootstrap=params['n_bootstrap'], seed=params['seed'],
                                       logdir=params['path'], logger=logger)


def main():
    parser = argparse.ArgumentParser(prog='Modular Goal Exploration with full information',
                                     description='This script performs a Modular Goal Exploration experiment')

    parser.add_argument('environment', help="The Environment you want to use", type=str,
                        choices=['armballs', 'bigarmballs'])
    parser.add_argument('representation', help="The Representation you want to use", type=str,
                        choices=["flat", "modular"])
    parser.add_argument('interest_model', help="The interest model you want to use", type=str,
                        choices=['uniform', 'normal', 'active'])

    parser.add_argument('--n_rbf', help="Number of RBF to use", type=int, default=7)
    parser.add_argument('--sdev', help="Standard deviation of RBF", type=float, default=10.)
    parser.add_argument('--object_size', help="Radius of the ball", type=float, default=0.17)
    parser.add_argument('--distract_size', help="Radius of the distractor", type=float, default=0.15)
    parser.add_argument('--distract_noise', help="Noise of the distractor", type=float, default=0.1)

    parser.add_argument('--n_explore', help="Number of exploration actions compared to exploitation actions",
                        type=int, default=4)
    parser.add_argument('--explo_ratio',
                        help="Proportion of exploration of modules with respect to choice proportional to interest",
                        type=float, default=0.1)
    parser.add_argument('--explo_noise_sdev', help="Noise added to exploration actions", type=float, default=.01)
    parser.add_argument('--win_size', help="Decay rate for computing interest", type=int, default=1000)
    parser.add_argument('--n_bootstrap', help="Number of bootstrapping actions", type=int, default=100)
    parser.add_argument('--n_exploration_iterations', help="Number of exploration iterations", type=int, default=int(1e4))

    parser.add_argument('--seed', help="Number of random motor babbling iterations", type=int, default=0)
    parser.add_argument('--path', help='Path to the results folder', type=str, default='.')
    parser.add_argument('--name', help='Name of the experiment', type=str, default='')
    parser.add_argument('-t', '--test', help='Whether to make a (shorter) test run', action="store_true")

    args = vars(parser.parse_args())

    assert os.path.isdir(args['path']), "You provided a wrong path."

    if args['name'] == '':
        args['name'] = ("MGE-FI %s %s" % (args['environment'], str(datetime.datetime.now()))).title()

    if args['test']:
        args['path'] = 'test'
    args['path'] = os.path.join(args['path'], args['name'])
    logger = logging.getLogger(args['name'], )
    logger.setLevel(logging.INFO)

    os.mkdir(args['path'])
    handler = logging.FileHandler(os.path.join(args['path'], 'logs.txt'))
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s[%(module)s:%(funcName)s:%(lineno)d]  %(message)s"))
    logger.addHandler(handler)

    run_experiments(args)


if __name__ == "__main__":
    main()