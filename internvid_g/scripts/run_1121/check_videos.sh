#!/bin/bash
#SBATCH --job-name=internvid
#SBATCH --qos=lv0b
#SBATCH -p HGX
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:0
#SBATCH --output=./code/nohup/internvid.log
#SBATCH --error=./code/nohup/internvid.log

python code/check_videos.py