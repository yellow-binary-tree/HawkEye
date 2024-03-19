#!/bin/bash
#SBATCH --job-name=of
#SBATCH --qos=lv4
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --output=./code/nohup/webdvd_of.out
#SBATCH --error=./code/nohup/webvid_of.out

# extract optical flow

cd /home/wangyuxuan1/codes/video_features

python main.py \
    feature_type=raft \
    device="cuda:0" \
    on_extraction=save_numpy \
    batch_size=4 \
    side_size=224 \
    file_with_video_paths=/scratch2/nlp/wangyueqian/InternVid/code/scripts/webvid_of_test_path.txt \
    output_path=/scratch2/nlp/wangyueqian/InternVid/code/webvid_of \
    # extraction_fps=2 