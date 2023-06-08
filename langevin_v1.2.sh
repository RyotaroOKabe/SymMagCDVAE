#!/bin/bash
#SBATCH --account=rokabe		# username to associate with job
#SBATCH --job-name=data		# a desired name to appear alongside job ID in squeue
#SBATCH --gres=gpu:1 			# number of GPUs (per node)
#SBATCH --time=3-23:00			# time (DD-HH:MM)
#SBATCH --output="slurm/%x_%j.out"		# output file where all text printed to terminal will be stored
					# current format is set to "job-name_jobID.out"
nice python langevin.py --model_path /home/rokabe/data2/generative/hydra/singlerun/2023-05-18/mp_20_1 --tasks gen --sg 1 --label n1a100 --alpha 100 --save_traj True
