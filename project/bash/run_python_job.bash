#!/bin/bash
#SBATCH -N 1  # number of  minimum nodes
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:1   # Request 1 gpu
#SBATCH -p 236606,all #  or all,236606, for more details see explanation above (srun command)
#SBATCH --mail-user=volodpol@cs.technion.ac.il
# (change to your own email if you wish to get one, or just delete this and the following lines)
#SBATCH --mail-type=ALL  # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name="just_a_test"
#SBATCH -o slurm.%N.%j.out    # stdout goes here
#SBATCH -e slurm.%N.%j.out   # stderr goes here

nvidia-smi
source ~/tensorflow/bin/activate
echo "Adding `pwd` to python environment"
export PYTHONPATH="`pwd`:${PYTHONPATH}"
python3 $@

