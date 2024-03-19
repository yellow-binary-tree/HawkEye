# construct grounding data from caption, scene, and scene sim data

import os
import json
import subprocess
import argparse

from tqdm import tqdm
import numpy as np


def parse_sec(sec_str):
    """
    Parse a string of the form '00:00:00.000' into a float
    """
    sec_str, ms_str = sec_str.split('.')
    h, m, s = sec_str.split(':')
    res = float(h) * 3600 + float(m) * 60 + float(s) + float(ms_str) / 1000
    return round(res, 3)


def find_scene_id_start(value, arr):
    # 大于等于value的最小的数的下标
    low, high = 0, len(arr) - 1
    result = 0
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] >= value:
            result = mid  # 更新结果为当前位置
            high = mid - 1  # 缩小搜索范围到左半部分
        else:
            low = mid + 1   # 扩大搜索范围到右半部分
    return result


def find_scene_id_end(value, arr):
    # 小于等于value的最大的数的下标
    low, high = 0, len(arr) - 1
    result = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] <= value:
            result = mid  # 更新结果为当前位置
            low = mid + 1  # 缩小搜索范围到左半部分
        else:
            high = mid - 1   # 扩大搜索范围到右半部分
    return result


def get_video_length(video_fname):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_fname],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)


def reformat_caption_data(caption_data):
    '''
    ret: {
        video_id: [{'start_sec': s1, 'end_sec': e1, 'captions': [c11, c12,. ...]}, ...],
        ...
    }
    '''
    ret = dict()
    for example in caption_data:
        key = example.get('YoutubeID', example.get('video_fname').split('.')[0])
        if key not in ret:
            ret[key] = list()
        if 'Start_timestamp' in example:
            example['start_sec'] = parse_sec(example['Start_timestamp'])
            del example['Start_timestamp']
        if 'End_timestamp' in example:
            example['end_sec'] = parse_sec(example['End_timestamp'])
            del example['End_timestamp']
        if 'Caption' in example:
            example['captions'] = [example['Caption']]
            del example['Caption']
        ret[key].append(example)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-base-folder", type=str, default="videos")
    parser.add_argument("--caption-fname", type=str)
    parser.add_argument("--scene-fname", type=str, default="temp/scenes-merged.jsonl")
    parser.add_argument("--scene-sim-fname", type=str, default="temp/scenes_similarity-merged.jsonl")
    parser.add_argument("--caption-with-neg-interval-fname", type=str, default="temp/final_dataset.jsonl")

    parser.add_argument("--scene-boundary-shift", type=float, default=0.25)
    parser.add_argument("--span-same-threshold", type=float, default=0.8)
    parser.add_argument("--min-caption-span-secs", type=float, default=0.5)     # about only 1% of the examples in caption.jsonl is shorter than 0.5 sec
    parser.add_argument("--max-caption-span-secs", type=float, default=10)     # if the span is too long, the caption may not precise, and the neg span can be very small. so this is a bad case for constructing grounding data
    parser.add_argument("--max-num-scenes", type=int, default=100)
    args = parser.parse_args()

    caption_data = [json.loads(line) for line in open(args.caption_fname)]
    caption_data = reformat_caption_data(caption_data)
    scene_data = [json.loads(line) for line in open(args.scene_fname)]
    video_id_to_fname = {e['video_fname'].split('.')[0]: e['video_fname'] for e in scene_data}
    scene_data = {e['video_fname'].split('.')[0]: e['scenes'][:args.max_num_scenes] for e in scene_data}
    scene_sim_data = [json.loads(line) for line in open(args.scene_sim_fname)]
    scene_sim_data = {e['video_fname'].split('.')[0]: e['similarity_matrix'] for e in scene_sim_data}

    video_ids = caption_data.keys() & scene_data.keys() & scene_sim_data.keys()
    print(len(video_ids))

    f_out = open(args.caption_with_neg_interval_fname, 'w')
    for i, video_id in enumerate(video_ids):
        video_length_sec = get_video_length(os.path.join(args.video_base_folder, video_id_to_fname[video_id]))
        # find the scene of caption segment
        for caption_data_by_id in caption_data[video_id]:
            caption_start_sec, caption_end_sec = caption_data_by_id['start_sec'], caption_data_by_id['end_sec']
            if caption_end_sec - caption_start_sec < args.min_caption_span_secs or caption_end_sec - caption_start_sec > args.max_caption_span_secs:
                continue
            caption_start_scene = find_scene_id_start(caption_start_sec - args.scene_boundary_shift, [span[0] for span in scene_data[video_id]])
            caption_end_scene = find_scene_id_end(caption_end_sec + args.scene_boundary_shift, [span[1] for span in scene_data[video_id]])

            unsimilar_start_scene, unsimilar_end_scene = None, None
            unsimilar_scenes = []
            if caption_end_scene < caption_start_scene:     # cannot find the scene of the caption segment
                caption_start_scene, caption_end_scene = None, None
                unsimilar_start_sec, unsimilar_end_sec = 0, video_length_sec

            else:
                # find if the neighbouring scene is also similar. if yes, merge them as the positive span
                new_caption_start_scene, new_caption_end_scene = None, None
                simiarity_matrix = np.array(scene_sim_data[video_id])
                for idx in range(caption_start_scene - 1, -1, -1):
                    if simiarity_matrix[idx, caption_start_scene] > args.span_same_threshold:
                        new_caption_start_scene = idx
                    else:
                        break
                if new_caption_start_scene is not None:
                    caption_start_scene = new_caption_start_scene
                    caption_start_sec = scene_data[video_id][caption_start_scene][0]

                for idx in range(caption_end_scene + 1, len(simiarity_matrix)):
                    if simiarity_matrix[idx, caption_end_scene] > args.span_same_threshold:
                        new_caption_end_scene = idx
                    else:
                        break
                if new_caption_end_scene is not None:
                    caption_end_scene = new_caption_end_scene
                    caption_end_sec = scene_data[video_id][caption_end_scene][1]

                # find the unsimilar scenes with the caption scenes
                scene_sims = np.max(simiarity_matrix[caption_start_scene: caption_end_scene + 1], axis=0)
                unsimilar_scenes = scene_sims < args.span_same_threshold

                if caption_start_scene == 0:
                    unsimilar_start_sec = 0
                elif np.all(unsimilar_scenes[:caption_start_scene]):  # no similar segment before this segment
                    unsimilar_start_sec = 0
                elif not np.any(unsimilar_scenes[:caption_start_scene]):  # all segments are similar segments before this segment
                    unsimilar_start_sec = caption_start_sec
                else:
                    # get the last similar segment
                    unsimilar_start_scene = int(caption_start_scene - np.argmin(unsimilar_scenes[caption_start_scene-1::-1]) - 1)
                    unsimilar_start_sec = scene_data[video_id][unsimilar_start_scene][1]

                if caption_end_scene == len(scene_data[video_id]) - 1:
                    unsimilar_end_sec = video_length_sec
                elif np.all(unsimilar_scenes[caption_end_scene+1:]):     # no similar segment after this segment
                    unsimilar_end_sec = video_length_sec
                elif not np.any(unsimilar_scenes[caption_end_scene+1:]):     # all segments are similar segments after this segment
                    unsimilar_end_sec = caption_end_sec
                else:
                    # get the first similar segment
                    unsimilar_end_scene = int(np.argmin(unsimilar_scenes[caption_end_scene+1:]) + caption_end_scene + 1)
                    unsimilar_end_sec = scene_data[video_id][unsimilar_end_scene][0]
                unsimilar_scenes = unsimilar_scenes.tolist()

            if unsimilar_end_scene == caption_end_scene and unsimilar_start_scene == caption_start_scene:
                continue # no neg interval found, do not use this example for grounding
            if caption_end_sec - caption_start_sec > args.max_caption_span_secs:
                continue

            for caption in caption_data_by_id['captions']:
                res_to_write = {'video': video_id_to_fname[video_id], 'duration': video_length_sec,
                                'start_sec': caption_start_sec, 'end_sec': caption_end_sec,
                                'neg_start_sec': unsimilar_start_sec, 'neg_end_sec': unsimilar_end_sec,
                                'caption': caption,
                                'start_scene': caption_start_scene, 'end_scene': caption_end_scene,
                                'neg_start_scene': unsimilar_start_scene, 'neg_end_scene': unsimilar_end_scene,}

                json.dump(res_to_write, f_out)
                f_out.write('\n')
        if i % 100 == 0:
            f_out.flush()
    f_out.close()
