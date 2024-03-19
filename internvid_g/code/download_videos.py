# download videos from youtube

import time
import os
import json
from multiprocessing import Pool
import argparse


def download_video(command):
    # time.sleep(2)
    video_id, target_dir = command
    # download videos with resolution nearest to 720p
    to_execute = f'yt-dlp "http://youtu.be/{video_id}" -o "{target_dir}/{video_id}.%(ext)s" -S "+res:720,fps" -q --no-check-certificate'
    res_code = os.system(to_execute)
    if res_code:
        error_video_fout.write(video_id + '\n')     # though this is not multi-threaded safe, but this is less important
        error_video_fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_dirs", type=str, nargs="+", default=["videos"], help="folders with videos already downloaded, so we dont need to download them again")
    parser.add_argument("--target_dir", type=str, default='videos', help="folder to save downloaded videos")
    parser.add_argument("--src_file", type=str, default='temp/video_ids.txt', help="folder that contains video ids to download")
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--error_file", type=str, default='temp/error_video_ids.txt')
    args = parser.parse_args()

    video_ids = set()
    if args.src_file.endswith('jsonl'):        # download from InternVid annotations or InternVid-G annotations
        for line in open(args.src_file):
            example = json.loads(line)
            if 'YoutubeID' in example:
                video_ids.add(example['YoutubeID'])
            if 'video_fname' in example:
                video_ids.add(example['video_fname'].split('.')[0])

    elif args.src_file.endswith('json'):        # download from InternVid annotations or InternVid-G annotations
        data = json.load(open(args.src_file))
        for example in data:
            video_ids.add(example['video'][:11])        # the first 11 letters is the youtube id

    elif args.src_file.endswith('txt'):        # download from InternVid annotations or InternVid-G annotations
        for line in open(args.src_file):
            video_ids.add(line.strip())

    else:       # change annotation loading code as you wish
        raise NotImplementedError

    error_video_ids = set([line.strip() for line in open(args.error_file)]) if os.path.exists(args.error_file) else set()
    videos_to_download = video_ids - error_video_ids

    for dir in args.old_dirs + [args.target_dir]:
        if not os.path.exists(dir):
            print('path does not exist, skipping stat video in this folder', dir)
            continue
        videos_downloaded = set(['.'.join(fname.split('.')[:-1]) for fname in os.listdir(dir)])
        videos_to_download = videos_to_download - videos_downloaded

    videos_to_download = list(videos_to_download)
    videos_to_download.sort()

    print(f'{len(video_ids)} videos in total')
    print(f'{len(error_video_ids)} videos error')
    print(f'{len(videos_downloaded)} videos already downloaded')
    print(f'{len(videos_to_download)} videos to download')

    error_video_fout = open(args.error_file, 'a')
    commands = [(video_id, args.target_dir) for video_id in videos_downloaded]
    with Pool(args.num_workers) as p:
        p.map(download_video, commands)
