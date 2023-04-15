import os
import subprocess

# write a script that runs main.py with all hyperparameters


# Part 1 Project
perc_wgts = [0.0, 0.01]
l1_wgts = [0.0, 10.0]
l2_wgts = [0.0, 0.001]
models = [('vanilla', 'z'), ('stylegan', 'z'), ('stylegan', 'w'), ('stylegan', 'w+')]
modes = ['project']

# Part 2 interpolate
perc_wgts = [0.01]
l1_wgts = [10.0]
l2_wgts = [0.0, 0.001]
models = [('vanilla', 'z'), ('stylegan', 'z'), ('stylegan', 'w'), ('stylegan', 'w+')]
modes = ['interpolate']

# Part 2 interpolate
perc_wgts = [0.01]
l1_wgts = [10.0]
l2_wgts = [0.001]
models = [('stylegan', 'w+')]
modes = ['draw']


for perc_wgt in perc_wgts:
    for l1_wgt in l1_wgts:
        for l2_wgt in l2_wgts:
            for (model, latent) in models:
                for mode in modes:
                    wild_str = ''
                    # wild_str = f"'./data/sketch/*.png'" if mode == 'draw' else ''
                    cmd = f'sbatch ./launch.sh --mode {mode} --perc_wgt {perc_wgt} --l1_wgt {l1_wgt} --l2_wgt {l2_wgt} --model {model} --latent {latent} {wild_str}'
                    subprocess.call(cmd, shell=True)