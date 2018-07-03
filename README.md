# Curiosity Driven Goal Exploration of Learned Disentangled Goal Spaces

This folder hosts the code to reproduce the results presented in the paper [Curiosity Driven Goal Exploration of Learned Disentangled Goal Spaces]. In this paper, we study the impact of the structure of the representation when it is used as a goal space in Intrinsically Motivated Goal Exploration Processes. Experiments are performed on a simple task in which multi-joint arm must handle and gather an oobject in a 2D space in the presence of a distractor which cannot be handled and follows a random walk.

## Running the experiments

To run a single experiment, you can run one of the three following python scripts:

+ `rpe.py` to perform a Random Parameterization Exploration
+ `mge_fi.py` to perform a Modular(Random) Goal Exploration using Engineered Features Representation
+ `mge_rep.py` to perform a Modular(Random) Goal Exploration using a learned Representation

Examples of some exploration algorithms together with a demonstration of the environment are provided in the notebook:

+ `notebooks/0_ArmBalls.ipynb`

In order to reproduce the results, you can also run a full campaign batch by running the following python scripts:
+ `script_rpe.py` to perform a Random Parameterization Exploration campaign
+ `script_mge_efr.py` to perform a Modular(Random) Goal Exploration using Engineered Features Representation campaign
+ `script_mge_rep.py` to perform a Modular(Random) Goal Exploration using a learned Representation campaign
and executing the generated scripts.

Alternatively, one can use the data provided in the file `results/armballs_dataset.pkl`

Finally, to generate the different figures out of the raw results, you can use the notebooks:

+ `results/Experiment_Visualization.ipynb` to compare the performances of the different algorithms or see the individual results.