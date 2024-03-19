#!/bin/bash
#SBATCH --job-name=internvid
#SBATCH --qos=lv4
#SBATCH -p HGX
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=./code/nohup/internvid.log
#SBATCH --error=./code/nohup/internvid.log

# for i in 0 1 2 3 4
# do
#     python -u code/clip_sim.py --func split_scene \
#         --video-sample-fname temp/1121/video_ids.txt \
#         --scene-fname temp/1121/scenes.jsonl.$i \
#         --start-idx $((i*1800)) --end-idx $((i*1800+1800)) \
#         > ./code/nohup/split_scene.log.$i 2>&1 &
# done
# wait

# cat temp/1121/scenes.jsonl.* > temp/1121/scenes.jsonl
# GPUS=(0 0 1 1 2 2)
# for i in 0 1 2 3 4 5
# do
#     CUDA_VISIBLE_DEVICES=${GPUS[i]} \
#     python code/clip_sim.py --func scene_sim \
#         --scene-fname temp/1121/scenes.jsonl \
#         --scene-sim-fname temp/1121/scenes_similarity.jsonl.$i \
#         --start-idx $((i*1500)) --end-idx $((i*1500+1500)) \
#         > ./code/nohup/scene_sim.log.$i 2>&1 &
# done 
# wait

# cat scenes_similarity.jsonl.* > scenes_similarity.jsonl
# python code/ground_data_construction.py \
#     --scene-fname temp/1121/scenes.jsonl --scene-sim-fname temp/1121/scenes_similarity.jsonl \
#     --caption-with-neg-interval-fname temp/1121/caption_with_neg_interval.jsonl


# merge scene
# python code/clip_sim.py --func merge_scene \
#     --scene-fname temp/1121/scenes.jsonl --scene-sim-fname temp/1121/scenes_similarity.jsonl \
#     --scene-merged-fname temp/1121/scenes_merged.jsonl --scene-merged-sim-fname temp/1121/scenes_merged_similarity.jsonl \

# scene captioning with blip2
# CUDA_VISIBLE_DEVICES=0 \
# python code/caption_clips.py --func blip2 \
#     --blip2-fname temp/scene_captions_blip2.jsonl

# filter similarity, though the summary now does not include blip2
# CUDA_VISIBLE_DEVICES=0 \
# python code/caption_clips.py --func filter \
#     --llama2-fname temp/scene_captions_llama2.jsonl \
#     --filtered-fname temp/scene_captions_blip_filtered.jsonl \
    # > code/nohup/scene_captions_filter.log 2>&1 &

