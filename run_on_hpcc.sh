#!/bin/sh

#SBATCH --account=guest
#SBATCH --partition=guest-compute
#SBATCH --qos=low
#SBATCH --time=24:00:00
#SBATCH --job-name=train_word2vec
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zacharyboroda@brandeis.edu
#SBATCH --output=trainingoutput/output_training_test.txt
#SBATCH --tasks=1
#SBATCH --cpus-per-task=128

module load share_modules/ANACONDA/5.3_py3

srun python training_test.py