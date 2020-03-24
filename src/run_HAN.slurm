#!/bin/bash 
#SBATCH -J He2
#SBATCH --gres=gpu:1  nvidia-smi
python main.py --lr 0.08
