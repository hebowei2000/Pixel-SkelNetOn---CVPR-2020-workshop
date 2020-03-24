#!/bin/bash 
#SBATCH -J HAN
#SBATCH --gres=gpu:1  nvidia-smi
python main.py
