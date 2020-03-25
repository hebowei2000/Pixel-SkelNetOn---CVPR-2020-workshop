#!/bin/bash 
#SBATCH -J He4
#SBATCH --gres=gpu:2  nvidia-smi
python main.py --lr 0.05
