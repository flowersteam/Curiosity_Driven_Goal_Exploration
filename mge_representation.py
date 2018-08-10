import argparse
import os
import logging
import datetime
import json

import numpy as np
import torch

from latentgoalexplo.actors import exploactors
from latentgoalexplo.environments import armballs
from latentgoalexplo.representation import representation_pytorch


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s[%(module)s:%(funcName)s:%(lineno)d]  %(message)s")
os.environ["JOBLIB_TEMP_FOLDER"] = "."


def ModularGoalExplorationUglExperiment(static_env, base_renderer, env_config, representation,
                                        network_type, n_channels, model_path, n_latents,
                                        interest_model, n_explore, explo_ratio, explo_noise_sdev, n_modules, win_size,
                                        n_bootstrap, n_exploration_iterations, s_bound,
                                        seed, logdir='test', log_interval=10, logger=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info("Observation phase")
    # We observe the ball moving (probably a scientist)
    a = base_renderer(**env_config)
    a.reset()

    training_images = []
    for i in range(1000):
        state = np.random.uniform(-1, 1, a.action_space.shape[0])
        a.act(observation=state)
        training_images.append(a.rendering)
    training_images = np.array(training_images)

    # We perform Bootstrap
    logger.info("Bootstrapping phase")
    a = exploactors.RandomParameterizationExploration(static_env=static_env, **env_config)
    a.reset()
    a.act(n_iter=n_bootstrap, render=False)

    logger.info("Loading representation")
    # We perform AGE-UGL
    b = exploactors.ActiveGoalExplorationUgl(static_env=static_env, representation=representation,
                                        network_type=network_type, n_channels=n_channels, n_latents=n_latents,
                                        beta=1, initial_epochs=0, learning_rate=0,
                                        interest_model=interest_model, n_explore=n_explore, explo_ratio=explo_ratio,
                                        n_modules=n_modules, explo_noise_sdev=explo_noise_sdev, win_size=win_size,
                                        s_bound=s_bound,
                                        visdom_record=False, log_interval=log_interval, **env_config)
    b.load_representation(actions=a.actions, outcomes=a.outcomes, outcomes_states=a.outcomes_states,
                          outcomes_train=training_images, model_path=model_path)
    logger.info("Starts exploration")
    b.act(n_iter=n_exploration_iterations, render=False)

    logger.info("Exploration over, saving data")
    # We save the representation
    attainable_representation = np.array(b.attainable_reps)
    np.save(os.path.join(logdir, 'attainable_representation'), attainable_representation.astype(np.float16))
    representation = np.array(b.outcomes_reps)
    np.save(os.path.join(logdir, 'outcomes_representation'), representation.astype(np.float16))
    sorted_latents = b._rep.sorted_latents
    np.save(os.path.join(logdir, 'sorted_latents'), sorted_latents.astype(np.float16))
    kld_latents = b._rep.kld_latents
    np.save(os.path.join(logdir, 'kld_latents'), kld_latents.astype(np.float16))
    # We save the set of explored states
    explored_states = np.array(b.outcomes_states)
    np.save(os.path.join(logdir, 'explored_states'), explored_states.astype(np.float32))
    interests_evolution = np.array(b.interests_evolution)
    np.save(os.path.join(logdir, 'interests_evolution'), interests_evolution.astype(np.float32))
    explo_evolution = np.array(b.explo_evolution)
    np.save(os.path.join(logdir, 'explo_evolution'), explo_evolution.astype(np.float32))
    # We save the set of goals states
    goals_states = np.array(b.goals_states)
    np.save(os.path.join(logdir, 'goals_states'), goals_states.astype(np.float16))


def run_experiments(params):
    logger = logging.getLogger(params['name'])

    logger.info("Instantiating the Environment")

    if params['environment'] == "armballs":
        static_env = armballs.MyArmBallsObserved
        base_renderer = armballs.ArmBallsRenderer
        params['distract_size'] = 0.15
        params['rgb'] = True
        env_config = dict()
        env_config.update({'arm_lengths': np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]),
                           'object_size': params['object_size'], 'distract_size': params['distract_size'],
                           'distract_noise': params['distract_noise'], 'env_noise': params['env_noise'],
                           'n_rbf': params['n_rbf'], 'sdev': params['sdev'],
                           'width': 64, 'height': 64, 'rgb': params['rgb'], 'n_timesteps': 50, 'render_arm': False,
                           'render': False})
        params['n_channels'] = 3
        params['network_type'] = 'cnn'
        if params['representation'] == "betavae":
            # object_size = 0.17, distract_size = 0.15
            representation = representation_pytorch.PytorchBetaVAERepresentation
            model_path = 'weights/ArmBalls_rgb_BallDistract'
        elif params['representation'] == "vae":
            representation = representation_pytorch.PytorchBetaVAERepresentation
            # object_size = 0.17, distract_size = 0.15
            model_path = 'weights/ArmBalls_rgb_BallDistract_ent'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if params['test']:
        params['n_exploration_iterations'] = int(1e2)

    with open(os.path.join(params['path'], 'config.json'), 'w') as f:
        json.dump(params, f, separators=(',\n', ': '))

    logger.info("Instantiating the Explorator")
    ModularGoalExplorationUglExperiment(static_env=static_env, base_renderer=base_renderer, env_config=env_config,
                                        representation=representation, model_path=model_path,
                                        network_type=params['network_type'],
                                        n_channels=params['n_channels'], n_latents=params['n_latents'],
                                        interest_model=params['interest_model'],
                                        n_explore=params['n_explore'], explo_ratio=params['explo_ratio'],
                                        n_modules=params['n_modules'], win_size=params['win_size'],
                                        n_bootstrap=params['n_bootstrap'], explo_noise_sdev=params['explo_noise_sdev'],
                                        n_exploration_iterations=params['n_exploration_iterations'], s_bound=params['s_bound'],
                                        seed=params['seed'], logdir=params['path'], logger=logger)


def main():
    parser = argparse.ArgumentParser(prog='Random Goal Babbling on Learned Goal Spaces',
                                     description='This script performs experiment on Unsupervised Goal Learning')

    parser.add_argument('environment', help="the Environment you want to use", type=str,
                        choices=['armballs'])
    parser.add_argument('representation', help="the Representation you want to use", type=str,
                        choices=['vae', 'betavae'])
    parser.add_argument('interest_model', help="The intereset model you want to use", type=str,
                        choices=['uniform', 'normal', 'active'])

    parser.add_argument('--n_rbf', help="Number of RBF to use", type=int, default=7)
    parser.add_argument('--sdev', help="Standard deviation of RBF", type=float, default=10.)
    parser.add_argument('--object_size', help="Radius of the ball", type=float, default=0.17)
    parser.add_argument('--distract_size', help="Radius of the distractor", type=float, default=0.15)
    parser.add_argument('--distract_noise', help="Radius of the distractor", type=float, default=0.1)
    parser.add_argument('--env_noise', help='Noise in the rendering of the environment (observation)', type=float,
                        default=0.)

    parser.add_argument('--n_latents', help="Number of latent dimensions to use", type=int, default=10)

    parser.add_argument('--n_explore', help="Number of exploration actions compared to exploitation actions",
                        type=int, default=4)
    parser.add_argument('--explo_ratio',
                        help="Proportion of exploration of modules with respect to choice proportional to interest",
                        type=float, default=0.1)
    parser.add_argument('--n_modules', help="Number of interests modules. Must be a fraction of the number of latents",
                        type=int, default=5)
    parser.add_argument('--explo_noise_sdev', help="Noise added to exploration actions", type=float, default=.01)
    parser.add_argument('--win_size', help="Decay rate for computing interest", type=int, default=1000)
    parser.add_argument('--s_bound', help="Bounds of sensori (latent) space", type=float, default=3.)
    parser.add_argument('--n_bootstrap', help="Number of bootstrapping actions", type=int, default=100)
    parser.add_argument('--n_exploration_iterations', help="Number of exploration iterations", type=int,
                        default=int(1e4))

    parser.add_argument('--seed', help="Number of random motor babbling iterations", type=int, default=0)
    parser.add_argument('--path', help='Path to the results folder', type=str, default='.')
    parser.add_argument('--name', help='Name of the experiment', type=str, default='')
    parser.add_argument('-t', '--test', help='Whether to make a (shorter) test run', action="store_true")

    args = vars(parser.parse_args())

    assert os.path.isdir(args['path']), "You provided a wrong path."

    if args['name'] == '':
        args['name'] = ("MGE-REP %s %s %s" % (
        args['representation'], args['environment'], str(datetime.datetime.now()))).title()

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
