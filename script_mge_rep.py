#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
import itertools

PATH_TO_RESULTS = "results/campaign/armballs"
PATH_TO_INTERPRETER = ""

envs = ['armballs']
reps = ['vae', 'betavae']
interest_models = ['normal']
n_modules = [1, 5]
object_sizes = [0.1]
distract_noises = [0., 0.1]
explo_noises = [0.05, 0.1]
n_bootstraps = [400]
exploration_iterations = [int(1e4)]
explo_ratios = [0.1]

params_iterator = list(itertools.product(envs, reps, interest_models, n_modules, object_sizes, distract_noises,
                                         explo_noises, n_bootstraps, exploration_iterations, explo_ratios))
nb_runs = 20

filename = 'campaign_{}.sh'.format(datetime.datetime.now().strftime("%d%m%y_%H%M"))
with open(filename, 'w') as f:
    f.write("export EXP_INTERP='%s' ;\n" % PATH_TO_INTERPRETER)
    for (env, rep, interest_model, n_module, object_size, distract_noise, explo_noise, n_bootstrap, exploration_iteration, explo_ratio) in params_iterator:
        f.write('ngpu="$(nvidia-smi -L | tee /dev/stderr | wc -l)"\n')
        f.write('agpu=0\n')
        for i in range(nb_runs):
            name = "MGE-REP_rep:{}_nmodules:{}_im:{}_env:{}_objectsize:{}_distract_noise:{}_explonoise:{}_date:{}".format(rep, n_module, interest_model, env, object_size, distract_noise, explo_noise, '$(date "+%d%m%y-%H%M-%3N")')
            f.write('echo "=================> %s";\n' % name)
            f.write('echo "=================> %s" >> log.txt;\n' % name)
            f.write('export CUDA_VISIBLE_DEVICES=$agpu\n')
            f.write("$EXP_INTERP mge_rep_armballs.py {env} {rep} {interest_model} --path={path} --name={name}"\
                    " --n_modules={n_module} --object_size={object_size} --distract_noise={distract_noise}"\
                    " --explo_noise_sdev={explo_noise} --explo_ratio={explo_ratio}"\
                    " --n_bootstrap={n_bootstrap} --n_exploration_iterations={exploration_iteration} --seed={seed}"\
                    " || (echo 'FAILURE' && echo 'FAILURE' >> log.txt) &\n".format(env=env,
                                                                                   rep=rep,
                                                                                   interest_model=interest_model,
                                                                                   n_module=n_module,
                                                                                   object_size=object_size,
                                                                                   distract_noise=distract_noise,
                                                                                   explo_noise=explo_noise,
                                                                                   n_bootstrap=n_bootstrap,
                                                                                   exploration_iteration=exploration_iteration,
                                                                                   explo_ratio=explo_ratio,
                                                                                   seed=i,
                                                                                   path=PATH_TO_RESULTS,
                                                                                   name='"{}"'.format(name)))
            f.write("agpu=$(((agpu+1)%ngpu))\n")
        f.write('wait\n')
