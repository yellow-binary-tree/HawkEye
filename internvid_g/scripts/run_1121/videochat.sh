#!/bin/bash
#SBATCH --job-name=i3-blip2-text2tag
#SBATCH --qos=lv4
#SBATCH -p HGX
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --output=./code/nohup/internvid.log
#SBATCH --error=./code/nohup/internvid.log

# for i in 0 # 1 2 3
# do
# CUDA_VISIBLE_DEVICES=$i python code/caption_clips.py --func videochat \
#     --video-folder videos-lowres --scene-fname temp/1121/scenes_merged.jsonl.0${i} \
#     --videochat-fname temp/1121/scene_captions_videochat.jsonl.${i} \
#     --videochat-config-fname code/scripts/run_1121/config_7b.json \
#     # > code/nohup/videochat.log.${i} 2>&1 &
# done
# wait
# 上面这个程序标注了一些内容,感觉经过finetune的videochat的特点就是会生成冗长但经常有幻觉的caption.
# 接下来还尝试了只经过pre-train的videochat,发现它生成的东西有很明显的webvid noise,即总是加上时间地点等,基本也不能用.

i=3
CUDA_VISIBLE_DEVICES=0 python code/caption_clips.py --func blip2 \
    --video-folder videos-lowres --scene-fname temp/1121/scenes_merged.jsonl.0${i} \
    --blip2-fname temp/1121/scene_captions_blip2.jsonl.${i} \
    >> code/nohup/blip2.log.${i} 2>&1 &

CUDA_VISIBLE_DEVICES=1 python code/caption_clips.py --func tag2text \
    --video-folder videos-lowres --scene-fname temp/1121/scenes_merged.jsonl.0${i} \
    --tag2text-fname temp/1121/scene_captions_tag2text.jsonl.${i} \
    >> code/nohup/tag2text.log.${i} 2>&1 &
wait