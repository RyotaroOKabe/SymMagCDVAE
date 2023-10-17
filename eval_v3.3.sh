#!/bin/bash
#SBATCH --account=rokabe		# username to associate with job
#SBATCH --job-name=eval		# a desired name to appear alongside job ID in squeue
#SBATCH --gres=gpu:1 			# number of GPUs (per node)
#SBATCH --time=3-23:00			# time (DD-HH:MM)
#SBATCH --output="slurm/%x_%j.out"		# output file where all text printed to terminal will be stored
					# current format is set to "job-name_jobID.out"
nice python scripts/evaluate.py --model_path /home/rokabe/data2/generative/hydra/singlerun/2023-09-07/mp20_1 \
	--label org --tasks recon gen opt --save_traj True \
	--n_step_each 50 --num_batches_to_sample 5 \
	--batch_size 50 --down_sample_traj_step 20