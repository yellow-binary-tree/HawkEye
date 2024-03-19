#!/bin/bash
#SBATCH --job-name=internvid
#SBATCH --qos=lv4
#SBATCH -p HGX
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:0
#SBATCH --output=./code/nohup/internvid.log
#SBATCH --error=./code/nohup/internvid.log

for i in 0 1 2 3
do
    python code/ground_data_construction.py \
        --video-base-folder videos-lowres \
        --caption-fname temp/1121/scene_captions_tag2text_high_sim.jsonl.$i \
        --scene-fname temp/1121/scenes_merged.jsonl.0$i \
        --scene-sim-fname temp/1121/scenes_merged_similarity.jsonl \
        --caption-with-neg-interval-fname temp/1121/scene_captions_tag2text_high_sim-with_neg.jsonl.$i &
done
wait