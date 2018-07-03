#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
import itertools

PATH_TO_RESULTS = "results/campaign/armballs"
PATH_TO_INTERPRETER = ""

envs = ['armballs']
object_sizes = [0.1]
distract_noises = [0., 0.1]
exploration_iterations = [int(1e4)]

params_iterator = list(itertools.product(envs, object_sizes, distract_noises, exploration_iterations))
nb_runs = 20

filename = 'campaign_{}.sh'.format(datetime.datetime.now().strftime("%d%m%y_%H%M"))
with open(filename, 'w') as f:
    f.write("export EXP_INTERP='%s' ;\n" % PATH_TO_INTERPRETER)
    for (env, object_size, distract_noise, exploration_iteration) in params_iterator:
        # f.write('ngpu="$(nvidia-smi -L | tee /dev/stderr | wc -l)"\n')
        # f.write('agpu=0\n')
        for i in range(nb_runs):
            name = "RPE_env:{}_objectsize:{}_distract_noise:{}_date:{}".format(env, object_size, distract_noise, '$(date "+%d%m%y-%H%M-%3N")')
            f.write('echo "=================> %s";\n' % name)
            f.write('echo "=================> %s" >> log.txt;\n' % name)
            f.write('export CUDA_VISIBLE_DEVICES=$agpu\n')
            f.write("$EXP_INTERP rpe.py {env} --path={path} --name={name}"\
                    " --object_size={object_size} --distract_noise={distract_noise}"\
                    " --n_exploration_iterations={exploration_iteration} --seed={seed}"\
                    " || (echo 'FAILURE' && echo 'FAILURE' >> log.txt) &\n".format(env=env,
                                                                                   object_size=object_size,
                                                                                   distract_noise=distract_noise,
                                                                                   exploration_iteration=exploration_iteration,
                                                                                   seed=i,
                                                                                   path=PATH_TO_RESULTS,
                                                                                   name='"{}"'.format(name)))
            # f.write("agpu=$(((agpu+1)%ngpu))\n")
        f.write('wait\n')
