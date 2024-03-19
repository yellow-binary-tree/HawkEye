#!/bin/bash
#SBATCH --job-name=filter-internvid
#SBATCH --qos=lv0b
#SBATCH -p HGX
#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=6
#SBATCH --gres=gpu:4
#SBATCH --output=./code/nohup/internvid.log
#SBATCH --error=./code/nohup/internvid.log


# 对llama2 summary之后的结果计算相似度
# for i in 0 1 2 3
# do
# CUDA_VISIBLE_DEVICES=$i python code/caption_clips.py --func filter \
#     --batch-size 32 \
#     --video-folder videos-lowres --scene-fname temp/1121/scenes_merged.jsonl.0${i} \
#     --llama2-fname temp/1121/scene_captions_llama2.jsonl.${i} \
#     --filtered-fname temp/1121/scene_captions_filtered.jsonl.${i} \
#     > code/nohup/clip_filter.log.${i} 2>&1 &
# done
# wait

# 对tag2text标注出的每个caption计算相似度
# for i in 0 1 2 3
# do
# CUDA_VISIBLE_DEVICES=$i python code/caption_clips.py --func filter \
#     --batch-size 32 \
#     --video-folder videos-lowres --scene-fname temp/1121/scenes_merged.jsonl.0${i} \
#     --filter-input-fname temp/1121/scene_captions_tag2text.jsonl.${i} --merge-method max \
#     --filtered-fname temp/1121/scene_captions_tag2text_clip_sim.jsonl.${i} \
#     > code/nohup/tag2text_clip_sim.log.${i} 2>&1 &
# done
# wait

# select tag2text captions with clip_sim score above median
for i in 0 1 2 3
do
    python code/caption_clips.py --func select_filtered_tag2text_captions \
        --filter-input-fname temp/1121/scene_captions_tag2text_clip_sim.jsonl.$i \
        --filtered-fname temp/1121/scene_captions_tag2text_high_sim.jsonl.$i
done
