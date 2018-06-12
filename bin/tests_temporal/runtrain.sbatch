#!/bin/sh
#SBATCH --mem=8gb --partition=gpu --gres=gpu --time=01:00:00 --output=output.txt
module load anaconda3
module load cuda
python RNN_Regression_Even_Duration.py