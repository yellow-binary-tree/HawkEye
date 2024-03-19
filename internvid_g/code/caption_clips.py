# Following InternVid, we densely caption some of the clips usiong tag2text 

import os
import math
import json
import argparse
from typing import Any
from PIL import Image

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import pipeline
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SceneLabelingDataset:
    def __init__(self, video_folder, scene_fname, transform=None, scene_min_secs=1, fps=1, max_num_frames=10, num_frames=None, start_idx=0, end_idx=None, output_fname=None):
        if num_frames is None:      # num frames and fps params are mutually exclusive
            assert 1/fps < scene_min_secs
        self.fps = fps
        self.video_folder = video_folder
        self.transform = transform if transform is not None else lambda x: x
        self.num_frames = num_frames
        self.max_num_frames = max_num_frames
        self.metadata = list()

        labeled_examples = set()
        if output_fname is not None and os.path.exists(output_fname):
            labeled_data = [json.loads(line) for line in open(output_fname)]
            for example in labeled_data:
                labeled_examples.add((example['video_fname'], example['start_sec'], example['end_sec']))

        for line_i, line in enumerate(open(scene_fname)):
            if line_i < start_idx: continue
            if line_i == end_idx: break
            example = json.loads(line)
            for start_sec, end_sec in example['scenes']:
                if end_sec - start_sec < scene_min_secs: continue
                if (example['video_fname'], start_sec, end_sec) in labeled_examples: continue
                self.metadata.append((example['video_fname'], start_sec, end_sec))

        print('loaded %d examples' % len(self))
        print('example data:', self[0])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        fname, start_sec, end_sec = self.metadata[idx]
        cap = cv2.VideoCapture(os.path.join(self.video_folder, fname))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if True:
            start_frame, end_frame = int(start_sec * fps), int(end_sec * fps)
            if self.num_frames is None:
                interval = max(math.ceil(fps / self.fps), math.ceil((end_frame - start_frame) / (self.max_num_frames + 1)))      # at most self.max_num_frames frames
            else:
                interval = math.ceil((end_frame - start_frame) / (self.num_frames + 1))
            frames_list = list()
            for frame_i in range(start_frame + interval, end_frame, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                ret, frame = cap.read()
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_list.append(self.transform(Image.fromarray(frame)))
            cap.release()
        return {'frame_list': frames_list, 'video_fname': fname, 'start_sec': start_sec, 'end_sec': end_sec}


class SceneCaptionSimDataset:
    def __init__(self, video_folder, scene_fname, processor=None, num_frames=3, clip_max_secs=None, output_fname=None):
        self.video_folder = video_folder
        self.preprocessor = processor        # CLIPProcessor
        self.num_frames = num_frames
        self.clip_max_secs = clip_max_secs

        labeled_examples = set()
        if output_fname and os.path.exists(output_fname):
            for line in open(output_fname):
                example = json.loads(line)
                labeled_examples.add((example['video_fname'], example['start_sec'], example['end_sec']))

        self.metadata = list()
        for line in open(scene_fname):
            example = json.loads(line)
            if (example['video_fname'], example['start_sec'], example['end_sec']) in labeled_examples: continue
            if self.clip_max_secs is not None and example['end_sec'] - example['start_sec'] > self.clip_max_secs:
                continue        # this clip is too long, possibily not useful at all

            caption = example.get('captions', example.get('summary', example.get('caption', list())))
            if isinstance(caption, str):
                if len(caption):
                    example['caption'] = caption
                    self.metadata.append(example)

            elif isinstance(caption, list):
                for cap in caption:
                    if len(cap):
                        example_new = example.copy()
                        example_new['caption'] = cap
                        self.metadata.append(example_new)

        print('loaded %d examples' % len(self))
        print('example data:', self[0])

    def __len__(self):
        return len(self.metadata)

    def load_frames(self, fname, start_sec, end_sec):
        cap = cv2.VideoCapture(fname)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame, end_frame = int(start_sec * fps), int(end_sec * fps)
        interval = (end_frame - start_frame) // (self.num_frames + 1)
        frames_list = list()
        # for frame_i in range(start_frame + interval, end_frame, interval):
        for i in range(self.num_frames):
            frame_i = start_frame + (i + 1) * interval
            # print(fname, fps, start_frame, end_frame, interval, frame_i)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
            ret, frame = cap.read()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(Image.fromarray(frame))
        cap.release()
        return frames_list

    def __getitem__(self, index):
        example = self.metadata[index]
        try:
            frames_list = self.load_frames(os.path.join(self.video_folder, example['video_fname']), example['start_sec'], example['end_sec'])
            pixel_values = self.preprocessor(images=frames_list, return_tensors='pt').pixel_values
            return {'is_valid': True, 'pixel_values': pixel_values, 'caption': example['caption'],
                    'video_fname': example['video_fname'], 'start_sec': example['start_sec'], 'end_sec': example['end_sec']}
        except:
            print('error when loading example:', example['video_fname'], example['start_sec'], example['end_sec'])
            return {'is_valid': False}


class SceneCaptionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.debug_print = False

    def __call__(self, examples):
        ret = dict()
        for key in examples[0]:
            lst = [e[key] for e in examples if e['is_valid']]
            if key == 'caption':
                ret['caption'] = lst
                ret['text_input'] = self.tokenizer(lst, truncation=True, padding='longest', return_tensors='pt')
            elif key == 'pixel_values':
                ret['visual_input'] = torch.cat(lst)
            else:
                ret[key] = lst
        
        # debug print
        if self.debug_print:
            self.debug_print = False
            for key, val in ret.items():
                if isinstance(val, torch.Tensor):
                    print(key, val.size())
                else:
                    print(key, val)
        return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='captioning')
    parser.add_argument('--func', type=str, default='tag2text')

    # this is used when the dataset is large, and you only process a fraction of it at a time
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=None)

    parser.add_argument('--video-folder', default='videos', type=str)
    parser.add_argument('--scene-fname', default='temp/scenes-merged.jsonl', type=str)
    parser.add_argument('--video-fps', default=2, type=float, metavar='N', help='sample fps of the video clip')
    parser.add_argument('--max-num-frames', default=5, type=int, metavar='N', help='max num frames to sample in a clip')
    parser.add_argument('--scene-min-secs', default=1, type=float, metavar='N', help='min length of scenes that are needed for captioning')

    # tag2text
    parser.add_argument('--pretrained',
                        metavar='DIR',
                        help='path to pretrained model',
                        default='/path/to/tag2text/tag2text_swin_14m.pth')
    parser.add_argument('--image-size',
                        default=384,
                        type=int,
                        metavar='N',
                        help='input image size')
    parser.add_argument('--thre',
                        default=0.68,
                        type=float,
                        metavar='N',
                        help='threshold value')
    parser.add_argument('--specified-tags',
                        default='None',
                        help='User input specified tags')
    parser.add_argument('--tag2text-fname', type=str)

    # blip2
    parser.add_argument('--blip2-ckpt', type=str, default='/scratch2/nlp/plm/blip2-flan-t5-xxl')
    parser.add_argument('--blip2-fname', type=str)

    # llama2
    parser.add_argument('--llama2-ckpt', type=str, default='/scratch2/nlp/plm/Llama-2-13b-chat-hf')
    parser.add_argument('--llama2-fname', type=str)

    # videochat
    parser.add_argument('--videochat-config-fname', type=str, default='/scratch2/nlp/plm/Llama-2-13b-chat-hf')
    parser.add_argument('--videochat-fname', type=str)

    # filter
    parser.add_argument('--clip-path', type=str, default='/scratch/nlp/wangyueqian/models/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff')
    parser.add_argument('--filter-input-fname', type=str)
    parser.add_argument('--filtered-fname', type=str)
    parser.add_argument('--merge-method', type=str, default='mean', choices=['mean', 'max'])
    parser.add_argument('--num-frames', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--clip-max-secs', type=int, default=None)
    parser.add_argument('--sim-threshold', type=float, default=0.8)

    # merge_filtered_captions
    # also use args.filter_input_fname and args.filtered_fname

    args = parser.parse_args()
    print(args)

    if args.func == 'tag2text':
        from recognize_anything.ram.models import tag2text
        from recognize_anything.ram import get_transform, inference_tag2text
        delete_tag_index = [127, 2961, 3351, 3265, 3338, 3355, 3359]

        # load tag2text model
        tag2text_model = tag2text(pretrained=args.pretrained, image_size=args.image_size,
            vit='swin_b', delete_tag_index=delete_tag_index)
        tag2text_model.threshold = args.thre  # threshold for tagging
        tag2text_model.eval()
        tag2text_model = tag2text_model.to(device)
        tag2text_transform = get_transform(image_size=args.image_size)

        dataset = SceneLabelingDataset(
            video_folder=args.video_folder, scene_fname=args.scene_fname, transform=tag2text_transform, max_num_frames=args.max_num_frames,
            scene_min_secs=args.scene_min_secs, fps=args.video_fps, output_fname=args.tag2text_fname
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3, collate_fn=lambda x: x[0])

        f_out = open(args.tag2text_fname, 'a')
        with torch.no_grad():
            for example_i, example in enumerate(dataloader):
                if len(example['frame_list']) == 0:
                    captions = list()
                else:
                    example['frame_list'] = torch.stack(example['frame_list'])
                    captions, tag_predict = tag2text_model.generate(example['frame_list'].to(device), tag_input=None, max_length=50, return_tag_predict=True)
                    captions = list(set(captions))
                res_to_write = {'video_fname': example['video_fname'], 'start_sec': example['start_sec'], 'end_sec': example['end_sec'], 'captions': captions}
                json.dump(res_to_write, f_out)
                f_out.write('\n')
                if example_i % 100 == 0:
                    f_out.flush()

    elif args.func == 'llama2':
        prompt = '''<s>[INST] <<SYS>>
Your task is to generate one sentence as the summary of a given list of sentences that desctibe a short video clip. Try to include all actions and objects mentioned. Summarize the caption list in at most 30 words with only 1 sentence.
<</SYS>>

%s [/INST]'''       # prompt suggeested by: https://huggingface.co/blog/llama2#how-to-prompt-llama-2 and https://github.com/facebookresearch/llama-recipes/blob/main/docs/inference.md#prompt-llama-2

        print('llama 2 prompt:')
        print(prompt)
        generator = pipeline(task='text-generation', model=args.llama2_ckpt, device_map="auto", model_kwargs={"load_in_8bit": False, 'torch_dtype': torch.float16,})

        video_fnames = set()
        tag2text_captions = dict()
        if os.path.exists(args.tag2text_fname):
            print('loading tag2text caption')
            for line in open(args.tag2text_fname):
                example = json.loads(line)
                tag2text_captions[(example['video_fname'], example['start_sec'], example['end_sec'])] = example['captions']
            print('loaded %d tag2text caption' % len(tag2text_captions))
            video_fnames = tag2text_captions.keys()
        else:
            print('no tag2text caption')

        blip2_captions = dict()
        if os.path.exists(args.blip2_fname):
            print('loading blip2 caption')
            for line in open(args.blip2_fname):
                example = json.loads(line)
                blip2_captions[(example['video_fname'], example['start_sec'], example['end_sec'])] = example['captions']
            print('loaded %d blip2 caption' % len(blip2_captions))
            if video_fnames:
                video_fnames &= blip2_captions.keys()
            else:
                video_fnames = blip2_captions.keys()
        else:
            print('no blip2 caption')

        summarized_video_fnames = set()
        if os.path.exists(args.llama2_fname):
            for line in open(args.llama2_fname):
                example = json.loads(line)
                summarized_video_fnames.add((example['video_fname'], example['start_sec'], example['end_sec']))

        video_fnames = video_fnames - summarized_video_fnames
        video_fnames = sorted(list(video_fnames))

        print('%d clip captions to summarize' % len(video_fnames))

        f_out = open(args.llama2_fname, 'a')
        for video_i, video_fname in enumerate(video_fnames):
            captions = blip2_captions.get(video_fname, []) + tag2text_captions.get(video_fname, [])
            if not captions:
                continue
            text_input = prompt % '. '.join(captions)
            summary = generator(text_input, max_length=500)[0]['generated_text']  # todo: minus text input
            summary = summary[len(text_input):].split('\n')[-1]
            print(captions, summary)
            json.dump({'video_fname': video_fname[0], 'start_sec': video_fname[1], 'end_sec': video_fname[2], 'summary': summary}, f_out)
            f_out.write('\n')

            if video_i % 20 == 0:
                f_out.flush()
        f_out.close()

    elif args.func == 'blip2':
        # generator = pipeline(task='visual-question-answering', model=args.blip2_ckpt, device_map="auto", model_kwargs={"load_in_8bit": False, 'torch_dtype': torch.float16, })
        model = Blip2ForConditionalGeneration.from_pretrained(args.blip2_ckpt, torch_dtype=torch.float16, device_map="cuda")        # or may cause OOM if load too many ckpts to CPU memory at a time
        model.to(device)
        model.eval()
        processor = Blip2Processor.from_pretrained(args.blip2_ckpt)

        dataset = SceneLabelingDataset(
            video_folder=args.video_folder, scene_fname=args.scene_fname, num_frames=1,
            scene_min_secs=args.scene_min_secs, output_fname=args.blip2_fname,
            start_idx=args.start_idx, end_idx=args.end_idx,
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3, collate_fn=lambda x: x[0])
        f_out = open(args.blip2_fname, 'a')
        for example_i, example in enumerate(dataloader):
            if not example['frame_list']:
                captions = []
            else:
                inputs = processor(images=example['frame_list'][0], text='Describe this image in detail.', return_tensors="pt")
                inputs = inputs.to(device).to(torch.float16)
                generated_ids = model.generate(**inputs)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                captions = [generated_text]

            res_to_write = {'video_fname': example['video_fname'], 'start_sec': example['start_sec'], 'end_sec': example['end_sec'], 'captions': captions}
            json.dump(res_to_write, f_out)
            f_out.write('\n')
            if example_i % 10 == 0:
                f_out.flush()

    elif args.func == 'filter':
        image_processor = CLIPImageProcessor.from_pretrained(args.clip_path)
        tokenizer = CLIPTokenizer.from_pretrained(args.clip_path)
        vision_model = CLIPVisionModelWithProjection.from_pretrained(args.clip_path)
        text_model = CLIPTextModelWithProjection.from_pretrained(args.clip_path)
        vision_model.eval()
        vision_model.to(device)
        text_model.eval()
        text_model.to(device)

        dataset = SceneCaptionSimDataset(args.video_folder, args.filter_input_fname, processor=image_processor, num_frames=args.num_frames, clip_max_secs=args.clip_max_secs, output_fname=args.filtered_fname)
        collator = SceneCaptionCollator(tokenizer)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=5, collate_fn=collator)

        f_out = open(args.filtered_fname, 'a')
        with torch.no_grad():
            for batch_i, batch in enumerate(dataloader):
                batch_size = len(batch['video_fname'])
                visual_input = batch['visual_input'].to(device)
                visual_features = vision_model(visual_input).image_embeds
                visual_features = visual_features.view(batch_size, args.num_frames, -1)
                text_input = batch['text_input'].to(device)
                text_features = text_model(**text_input).text_embeds 
                text_features = text_features.unsqueeze(1).repeat(1, args.num_frames, 1)

                if visual_features.size(-1) != text_features.size(-1):
                    # 我也不知道为什么，但有时就是会这样
                    print('error in feature shape during batch', batch['video_fname'], batch['start_sec'], batch['end_sec'])
                    continue

                sims = F.cosine_similarity(visual_features, text_features, dim=-1)
                if args.merge_method == 'mean':
                    sims = torch.mean(sims, dim=1)
                elif args.merge_method == 'max':
                    sims = torch.max(sims, dim=1)[0]

                for sim_i, sim in enumerate(sims.cpu().tolist()):
                    # if sim > args.sim_threshold:
                    if True:        # save all clip sim scores, as we dont know how large the score is to indicate a good vision-text pair.
                        res = {k: batch[k][sim_i] for k in ['video_fname', 'start_sec', 'end_sec', 'caption']}
                        res['clip_sim'] = sim
                        json.dump(res, f_out)
                        f_out.write('\n')

                if batch_i % 100 == 0:
                    f_out.flush()
        f_out.close()

    elif args.func == 'videochat':
        # use videochat to caption the video clips. hope this can keep precise motions in the video
        import sys
        sys.path.append('./code/video_chat')
        from video_chat.utils.config import Config
        from video_chat.utils.easydict import EasyDict
        from video_chat.models.videochat import VideoChat
        from video_chat.demo import init_model
        from video_chat.conversation import Chat
        from video_chat.models.video_transformers import (
            GroupNormalize, GroupScale, GroupCenterCrop, 
            Stack, ToTorchFormatTensor
        )

        class ChatForClipLabeling(Chat):
            def upload_video(self, image, conv, img_list, num_segments):
                # if isinstance(image, str):
                # if isinstance(image, list) and isinstance(image[0], Image):
                if True:
                    vid_chat, msg = self.load_video(image, num_segments=num_segments, return_msg=True)
                    TC, H, W = vid_chat.shape
                    image = vid_chat.reshape(1, TC//3, 3, H, W).to(self.device)
                else:
                    raise NotImplementedError
                # print("Input video shape:", vid_chat.shape)
                image_emb, _ = self.model.encode_img(image)
                img_list.append(image_emb)
                conv.messages.append([
                    conv.roles[0], 
                    f"<Video><VideoHere></Video> {msg}\n"
                ])
                msg = "Received."
                # self.conv.append_message(self.conv.roles[1], msg)
                return msg, img_list, conv

            def load_video(self, video, num_segments=8, return_msg=False):
                # vr = VideoReader(video_path, ctx=cpu(0))
                # num_frames = len(vr)
                # frame_indices = self.get_index(num_frames, num_segments)
                
                # duration = len(vr) // vr.get_avg_fps()
                # index = np.linspace(0, len(vr)-1, num=int(duration))
                # buffer = vr.get_batch(index).asnumpy()
                images_group = video
                # transform
                input_mean = [0.48145466, 0.4578275, 0.40821073]
                input_std = [0.26862954, 0.26130258, 0.27577711]
                
                transform = T.Compose([
                    GroupScale(int(224), interpolation=InterpolationMode.BICUBIC),
                    GroupCenterCrop(224),
                    Stack(),
                    ToTorchFormatTensor(),
                    GroupNormalize(input_mean, input_std) 
                ])
                frame_indices, sec = [0, 1, 2, 3, 4, 5, 6, 7], 1
                # images_group = list()
                # for frame in buffer:
                #       img = Image.fromarray(frame)
                #       images_group.append(img)
                torch_imgs_224 = transform(images_group)
                if return_msg:
                    # msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
                    msg = ''        # no prompt for pre-trained model
                    return torch_imgs_224, msg
                else:
                    return torch_imgs_224

        print('Initializing VideoChat')
        cfg = Config.from_file(args.videochat_config_fname)
        model = VideoChat(config=cfg.model)
        model = model.to(torch.device(cfg.device))
        model = model.eval()
        chat = ChatForClipLabeling(model)

        dataset = SceneLabelingDataset(
            video_folder=args.video_folder, scene_fname=args.scene_fname, num_frames=8,     # 8: suggested num_frames for videochat
            scene_min_secs=args.scene_min_secs, output_fname=args.videochat_fname
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3, collate_fn=lambda x: x[0])
        f_out = open(args.videochat_fname, 'w')

        for example_i, example in enumerate(dataloader):
            chat_state = EasyDict({
                "system": "",
                # "roles": ("Human", "Assistant"),
                "roles": ("", ""),
                "messages": [],
                # "sep": "###"
                "sep": ""
            })
            # user_message = 'What is the content of this video?'
            user_message = ''
            img_list = []
            llm_message, img_list, chat_state = chat.upload_video(example['frame_list'], chat_state, img_list, num_segments=8)
            chat_state =  chat.ask(user_message, chat_state)
            llm_message, llm_message_token, chat_state = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=100)
            print('chat_state:', chat_state)        # check model input
            llm_message = llm_message.replace("<s>", "") # handle <s>
            res_to_write = {'video_fname': example['video_fname'], 'start_sec': example['start_sec'], 'end_sec': example['end_sec'], 'captions': [llm_message]}
            json.dump(res_to_write, f_out)
            f_out.write('\n')
            if example_i % 10 == 0:
                f_out.flush()

    elif args.func == 'select_filtered_captions':
        f_in = open(args.filter_input_fname, 'r')
        f_out = open(args.filtered_fname, 'w')
        prev_key, captions = tuple(), set()

        examples = [json.loads(line) for line in f_in]
        mean_sim_score = np.median([e['clip_sim'] for e in examples])

        for example in examples:
            if example['clip_sim'] < mean_sim_score:
                continue
            key = (example['video_fname'], example['start_sec'], example['end_sec'])
            if key != prev_key:
                if captions:
                    res_to_write = {'video_fname': prev_key[0], 'start_sec': prev_key[1], 'end_sec': prev_key[2], 'captions': list(captions)}
                    f_out.write(json.dumps(res_to_write) + '\n')
                prev_key = key
                captions = set()
            captions.add(example['caption'])

        if captions:
            res_to_write = {'video_fname': prev_key[0], 'start_sec': prev_key[1], 'end_sec': prev_key[2], 'captions': list(captions)}
            f_out.write(json.dumps(res_to_write) + '\n')

        f_in.close()
        f_out.close()

    else:
        raise NotImplementedError()