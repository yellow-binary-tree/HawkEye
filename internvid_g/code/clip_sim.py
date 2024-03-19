import os
import cv2
import math
import json
import argparse
import random
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

from scenedetect import detect, ContentDetector
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def round_numbers(obj, decimals=3):
    if isinstance(obj, list):
        return [round_numbers(item, decimals) for item in obj]
    elif isinstance(obj, dict):
        return {key: round_numbers(value, decimals) for key, value in obj.items()}
    elif isinstance(obj, (int, float)):
        return round(obj, decimals)
    else:
        return obj


def split_scene(fname, min_secs=1):
    content_detector = ContentDetector()
    scene_list = detect(fname, content_detector)
    scene_list = [[scene[0].get_seconds(), scene[1].get_seconds()] for scene in scene_list]
    scene_list = [[start_sec, end_sec] for start_sec, end_sec in scene_list if end_sec - start_sec >= min_secs]
    return scene_list


class CLIPFrameDataset:
    def __init__(
            self, video_folder, scene_fname, frames_per_scene=3, output_fname=None,
            start_idx=0, end_idx=None):
        self.video_folder = video_folder
        self.frames_per_scene = frames_per_scene
        self.max_num_scenes = 100
        self.metadata = list()

        labeled_fnames = set()
        if output_fname is not None and os.path.exists(output_fname):
            labeled_fnames = set([json.loads(line)['video_fname'] for line in open(output_fname)])

        self.metadata = list()
        for i, line in enumerate(open(scene_fname)):
            if i < start_idx: continue
            if i == end_idx: break
            example = json.loads(line)
            if example['video_fname'] in labeled_fnames:
                continue
            self.metadata.append(example)

        print('loaded %d examples' % len(self))
        print('example data:', self[0])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        try:
            data = self.metadata[index]
            fname, scenes = os.path.join(self.video_folder, data['video_fname']), data['scenes']
            cap = cv2.VideoCapture(fname)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_list = list()
            for start_sec, end_sec in scenes[:self.max_num_scenes]:
                start_frame, end_frame = int(start_sec * fps), int(end_sec * fps)
                interval = math.floor((end_frame - start_frame) / (self.frames_per_scene + 1)) + 1
                for i in range(1, self.frames_per_scene + 1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + interval * i)
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_list.append(frame)
            cap.release()
            return {'frame_list': frame_list, 'video_fname': data['video_fname']}
        except:
            print("error when reading video %s" % data['video_fname'])
            return None


def clip_encode(processor, model, frame_list):
    inputs = processor(images=frame_list, return_tensors="pt").to(device)
    outputs = model(**inputs)
    image_features = outputs.image_embeds
    return image_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default='split_scene')

    parser.add_argument('--video-sample-fname', type=str, default='temp/video_ids.txt')

    parser.add_argument('--video-base-folder', type=str, default='videos')
    parser.add_argument('--clip-path', type=str, default='clip-vit-large-patch14')
    parser.add_argument('--frames-per-scene', type=int, default=3)

    parser.add_argument('--clip-sim-threshold', type=float, default=0.85)
    parser.add_argument('--clip-min-secs', type=float, default=0.5)
    parser.add_argument('--same-scene-update-method', type=str, default='last', choices=['last', 'union'])

    parser.add_argument('--scene-fname', type=str, default='temp/scenes.jsonl')
    parser.add_argument('--scene-sim-fname', type=str, default='temp/scenes_similarity.jsonl')
    parser.add_argument('--scene-merged-fname', type=str, default='temp/scenes_merged.jsonl')
    parser.add_argument('--scene-merged-sim-fname', type=str, default='temp/scenes_merged_similarity.jsonl')

    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=None)

    args = parser.parse_args()
    print(args)

    # --------------------------------------------------
    # split all videos into scenes
    if args.func == 'split_scene':
        
        if os.path.exists(args.scene_fname):
            processed_fnames = set([json.loads(line)['video_fname'] for line in open(args.scene_fname)])
        else:
            processed_fnames = set()

        f_out = open(args.scene_fname, 'a')
        fnames = sorted([line.strip() for line in open(args.video_sample_fname)])[args.start_idx: args.end_idx]

        for i, fname in enumerate(fnames):
            if not os.path.isfile(os.path.join(args.video_base_folder, fname)): continue
            if fname in processed_fnames: continue
            try:
                split_scene_res = split_scene(os.path.join(args.video_base_folder, fname), min_secs=args.clip_min_secs)
                res_to_write = {'video_fname': fname, 'scenes': split_scene_res}
                json.dump(round_numbers(res_to_write), f_out)
                f_out.write('\n')
            except:
                print('error in %s' % fname)
            if i % 10 == 0:
                f_out.flush()
        f_out.close()

    # --------------------------------------------------
    # check CLIP similarity between all clips in a video
    elif args.func == 'scene_sim':
        processor = CLIPProcessor.from_pretrained(args.clip_path)
        model = CLIPVisionModelWithProjection.from_pretrained(args.clip_path)
        model.to(device)
        model.eval()

        # load frames for each scene from the video, and 
        dataset = CLIPFrameDataset(args.video_base_folder, args.scene_fname, args.frames_per_scene, args.scene_sim_fname, 
                                start_idx=args.start_idx, end_idx=args.end_idx)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3, collate_fn=lambda x: x[0])

        f_out = open(args.scene_sim_fname, 'a')
        with torch.no_grad():
            for example_i, example in enumerate(dataloader):
                try:
                    if not example['frame_list']:
                        print('no scene in video', example['video_fname'])
                        res_to_write = {'video_fname': example['video_fname'], 'similarity_matrix': []}
                    else:
                        features = clip_encode(processor, model, example['frame_list'])
                        num_frames, hidden_size = features.size()
                        clip_features = features.reshape(num_frames // args.frames_per_scene, args.frames_per_scene, hidden_size).mean(dim=1)
                        similarity_matrix = F.cosine_similarity(clip_features.unsqueeze(1), clip_features.unsqueeze(0), dim=2)
                        res_to_write = {'video_fname': example['video_fname'], 'similarity_matrix': similarity_matrix.cpu().numpy().tolist()}
                    json.dump(round_numbers(res_to_write), f_out)
                    f_out.write('\n')
                except:
                    pass

                if example_i % 100 == 0:
                    f_out.flush()
        f_out.close()

    # --------------------------------------------------
    # merge similar consecutive segments, get merged scene boundaries and sims
    elif args.func == 'merge_scene':
        f_out = open(args.scene_merged_sim_fname, 'w')
        scene_merge_spans = dict()
        for line_i, line in tqdm(enumerate(open(args.scene_sim_fname))):
            try:
                example = json.loads(line)
            except:
                print("error when loading line %d" % line_i)
                continue
            arr = np.array(example['similarity_matrix'])
            same_scene_dict = dict()
            indices = np.argwhere((arr > args.clip_sim_threshold) & (~np.eye(arr.shape[0], dtype=bool))).tolist()
            for i, j in indices:
                if i >= j: 
                    continue
                if i not in same_scene_dict:
                    same_scene_dict[i] = set()
                same_scene_dict[i].add(j)

            scene_i = 0
            spans = list()
            span, same_scenes = [], set()
            while scene_i < arr.shape[0]:
                if scene_i not in same_scenes and len(span):
                    spans.append([span[0], span[-1]])
                    span, same_scenes = [], set()

                span.append(scene_i)
                if args.same_scene_update_method == 'union':
                    same_scenes |= same_scene_dict.get(scene_i, set())
                elif args.same_scene_update_method == 'last':
                    same_scenes = same_scene_dict.get(scene_i, set())
                scene_i += 1

            if scene_i not in same_scenes and len(span):
                spans.append((span[0], span[-1]))
            scene_merge_spans[example['video_fname']] = spans

            # merge scene similraity matrix
            new_simimarity_matrix1 = np.zeros((len(spans), arr.shape[0]))
            new_simimarity_matrix2 = np.zeros((len(spans), len(spans)))
            for i, (start, end) in enumerate(spans):
                new_simimarity_matrix1[i, :] = arr[start: end+1, :].max(axis=0)
            for i, (start, end) in enumerate(spans):
                new_simimarity_matrix2[:, i] = new_simimarity_matrix1[:, start: end+1].max(axis=1)
            json.dump({'video_fname': example['video_fname'], 'similarity_matrix': new_simimarity_matrix2.tolist()}, f_out)
            f_out.write('\n')

            if line_i % 100 == 0:
                f_out.flush()

        f_out = open(args.scene_merged_fname, 'w')
        for i, line in tqdm(enumerate(open(args.scene_fname))):
            example = json.loads(line)
            if example['video_fname'] not in scene_merge_spans:     # no scene similarity data
                continue

            merged_start_ends = list()
            for start, end in scene_merge_spans[example['video_fname']]:
                merged_start_ends.append([example['scenes'][start][0], example['scenes'][end][1]])
            json.dump({'video_fname': example['video_fname'], 'scenes': merged_start_ends}, f_out)
            f_out.write('\n')
            if i % 100 == 0:
                f_out.flush()
        f_out.close()

    else:
        raise NotImplementedError()
