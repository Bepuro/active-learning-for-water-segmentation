#!/bin/sh



srun -n1 -p hiperf --nodelist=tesla-a100 --gres=gpu:1 --cpus-per-task=12 --mem=100000   -t 12:00:00 --job-name=active-learn python ./train_old.py ./config_landcover.py