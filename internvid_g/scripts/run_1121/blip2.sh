#!/bin/bash
#SBATCH --job-name=internvid
#SBATCH --qos=lv0b
#SBATCH -p HGX
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --output=./code/nohup/internvid.log
#SBATCH --error=./code/nohup/internvid.log

for i in 0 1 2 3
do
CUDA_VISIBLE_DEVICES=$i python code/caption_clips.py --func blip2 \
    --video-folder videos-lowres --scene-fname temp/1121/scenes_merged.jsonl.0${i} \
    --blip2-fname temp/1121/scene_captions_blip2.jsonl.${i} \
    > code/nohup/blip2.log.${i} 2>&1 &
done
wait