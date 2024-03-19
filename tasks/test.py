import sys
sys.path.append('.')

import os
import re
import json
from tqdm import tqdm, trange
import argparse

import cv2
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from IPython.display import Video, HTML

from peft import get_peft_model, LoraConfig, TaskType
import copy

from transformers import StoppingCriteria, StoppingCriteriaList

from PIL import Image
import numpy as np
from decord import VideoReader, cpu

from utils.config import Config
from models.hawkeye_it import HawkEye_it
from utils.easydict import EasyDict
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + ": " + message + conv.sep
        else:
            ret += role + ":"
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret


def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]

        if model.use_lora:
            seg_embs = [model.llama_model.base_model.model.model.embed_tokens_(seg_t) for seg_t in seg_tokens]
        else:
            seg_embs = [model.llama_model.model.embed_tokens_(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text + '\n'])
        

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([835]).to("cuda:0"),
        torch.tensor([2277, 29937]).to("cuda:0")]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    conv.messages[-1][1] = output_text
    if print_res:
        print(output_token)
        print(output_text)
    return output_text, output_token.cpu().numpy()


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg
    else:
        return torch_imgs


def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table


class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224, return_msg=False):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data,
                })

        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }

        self.num_segments = num_segments
        self.return_msg = return_msg

        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
        print("loaded %d data" % len(self))
        print("example data:", self.get_example_data())

    def get_example_data(self, index=0):
        data = self[index]
        return {k: v.size() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        video_secs = len(vr) / fps

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)

        if self.return_msg:
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return torch_imgs, msg, video_secs
        return torch_imgs, '', video_secs

    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        video_secs = len(gif) / fps

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        torch_imgs = self.transform(images_group)

        if self.return_msg:
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return torch_imgs, msg, video_secs
        return torch_imgs, '', video_secs

    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        video_secs = max_frame / fps

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) # frame_idx starts from 1
        fname_list = sorted(os.listdir(video_path))
        for frame_index in frame_indices:
            fname = fname_list[frame_index]
            # img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            img = Image.open(os.path.join(video_path, fname))
            images_group.append(img)
        torch_imgs = self.transform(images_group)

        if self.return_msg:
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return torch_imgs, msg, video_secs
        return torch_imgs, '', video_secs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data.get('answer', None)
        answer_idx = -1

        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        if answer is not None:
            answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        else:
            answer = '(x) None'
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound'] and 'start' in self.data_list[idx]['data'] and 'end' in self.data_list[idx]['data']:
            bound = (self.data_list[idx]['data']['start'], self.data_list[idx]['data']['end'])
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs, video_msg, video_secs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video_fname': self.data_list[idx]['data']['video'], 
            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'span': self.data_list[idx]['data'].get('span', None),
            'video_msg': video_msg,
            'video_secs': video_secs,
            'task_type': self.data_list[idx]['task_type']
        }


class GroundingDataset(MVBench_dataset):
    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        answer = data['answer']
        return question, answer


def generate_no_prompt(model, img_list, question,
                       do_sample=True, num_beams=1, min_length=1, top_p=0.9, max_new_tokens=100, 
                       repetition_penalty=1.0, length_penalty=1, temperature=1.0, print_res=False):
    stop_words_ids = [
        torch.tensor([835]).to("cuda:0"),
        torch.tensor([2277, 29937]).to("cuda:0")]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    if print_res:
        print(question)

    prompt_segs = question.split('<VideoHere>')
    with torch.no_grad():
        seg_tokens = [
            model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]

        if model.use_lora:
            seg_embs = [model.llama_model.base_model.model.model.embed_tokens_(seg_t) for seg_t in seg_tokens]
        else:
            seg_embs = [model.llama_model.model.embed_tokens_(seg_t) for seg_t in seg_tokens]

    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    embs = torch.cat(mixed_embs, dim=1)

    with torch.no_grad():
        outputs = model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )

    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    if print_res:
        print(output_token)
        print(output_text)
    return output_text


def infer_mvbench(
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        no_prompt=False,
        no_video=False,
    ):
    video = data_sample["video"]
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to("cuda:0")
    
    video_list = []
    with torch.no_grad():
        if system_q:
            video_emb, _ = model.encode_img(video, system + data_sample['question'])
        else:
            video_emb, _ = model.encode_img(video, system)
    video_list.append(video_emb)

    if no_prompt:
        question = data_sample['question'] if no_video else '<Video><VideoHere></Video>' + data_sample['question']
        llm_message = generate_no_prompt(model, video_list, question + answer_prompt, print_res=print_res)
    else:
        chat = EasyDict({
            "system": system,
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })

        if not args.no_video:
            chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> %s\n" % data_sample['video_msg'].rstrip()])

        if system_llm:
            prompt = system + data_sample['question'] + question_prompt
        else:
            prompt = data_sample['question'] + question_prompt
        
        ask(prompt, chat)

        with torch.cuda.amp.autocast():
            llm_message = answer(
                conv=chat, model=model, do_sample=False,
                img_list=video_list, max_new_tokens=100,
                answer_prompt=answer_prompt, print_res=print_res
            )[0]
    # remove potential explanation
    # llm_message = return_prompt + llm_message.strip().split('\n')[0]
    # sometimes we need this explanation
    llm_message = return_prompt + llm_message
    return llm_message


def check_ans(pred, gt_answer, gt_span=None, video_secs=None, num_frames=None, mode='choice'):
    if mode in ['choice', 'grounding_choice']:
        flag = False
        pred_list = pred.lower().split(' ')
        pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
        gt_list = gt_answer.lower().split(' ')
        gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
        if gt_content[-1] == '.':
            gt_content = gt_content[:-1]
        
        if pred_option.replace('.', '') in gt_option:
            flag = True
        elif gt_option in pred_option:
            flag = True
            
        return flag

    if mode in ['grounding_frame', 'grounding_sec']:       # is grounding task
        decimal_pattern = r'\d+\.\d+'
        gold_matches = re.findall(decimal_pattern, gt_answer)

        if mode == 'grounding_frame':
            integer_pattern = r'\d+'
            pred_matches = re.findall(integer_pattern, pred)[:2]
            if len(pred_matches) < 2: return None
            pred_matches = [min(num_frames, max(0, int(i))) for i in pred_matches]
            pred_matches = [int(pred_matches[0]) / num_frames * video_secs, (int(pred_matches[1]) + 1) / num_frames * video_secs]
        elif mode == 'grounding_sec':
            pred_matches = re.findall(decimal_pattern, pred)[:2]
            if len(pred_matches) < 2: return None

        gold_start, gold_end, pred_start, pred_end = float(gold_matches[0]), float(gold_matches[1]), float(pred_matches[0]), float(pred_matches[1])
        intersection = max(0, min(gold_end, pred_end) - max(gold_start, pred_start))
        union = max(0, max(gold_end, pred_end) - min(gold_start, pred_start))
        if union <= 0 or intersection <= 0:
            return 0
        return intersection / union


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.json")
    parser.add_argument("--ckpt", type=str, default="model/VideoChat2/videochat2_7b_stage3.pth")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--mode", type=str, default='choice')
    parser.add_argument("--save-path", type=str, default="test.json")
    parser.add_argument("--print-res", type=int, default=0)
    parser.add_argument("--no-video", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--return-video-msg", type=int, default=1)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--num-frame-tokens", type=int, default=0)
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--question-prompt", type=str, default=None)
    parser.add_argument("--answer-prompt", type=str, default=None)
    parser.add_argument("--return-prompt", type=str, default=None)
    parser.add_argument("--no-prompt", action='store_true')
    parser.add_argument("--tasks", type=str, nargs='+', default=["MVBench"])
    args = parser.parse_args()

    if args.no_prompt:
        args.return_video_msg = 0

    print(args)

    data_list = {
        "Moving Direction": (f"{args.data_dir}/MVBench/json/moving_direction.json", f"{args.data_dir}/MVBench/video/clevrer/video_validation/", "video", False),
        "Action Localization": (f"{args.data_dir}/MVBench/json/action_localization.json", f"{args.data_dir}/MVBench/video/sta/sta_video/", "video", True),  # has start & end
        "Scene Transition": (f"{args.data_dir}/MVBench/json/scene_transition.json", f"{args.data_dir}/MVBench/video/scene_qa/video/", "video", False),
        "Action Count": (f"{args.data_dir}/MVBench/json/action_count.json", f"{args.data_dir}/MVBench/video/perception/videos/", "video", False),
        "Moving Count": (f"{args.data_dir}/MVBench/json/moving_count.json", f"{args.data_dir}/MVBench/video/clevrer/video_validation/", "video", False),
        "Moving Attribute": (f"{args.data_dir}/MVBench/json/moving_attribute.json", f"{args.data_dir}/MVBench/video/clevrer/video_validation/", "video", False),
        "State Change": (f"{args.data_dir}/MVBench/json/state_change.json", f"{args.data_dir}/MVBench/video/perception/videos/", "video", False),
        "Fine-grained Pose": (f"{args.data_dir}/MVBench/json/fine_grained_pose.json", f"{args.data_dir}/MVBench/video/nturgbd/", "video", False),
        "Action Sequence": (f"{args.data_dir}/MVBench/json/action_sequence.json", f"{args.data_dir}/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
        "Action Prediction": (f"{args.data_dir}/MVBench/json/action_prediction.json", f"{args.data_dir}/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
        "Action Antonym": (f"{args.data_dir}/MVBench/json/action_antonym.json", f"{args.data_dir}/MVBench/video/ssv2_video/", "video", False),
        "Fine-grained Action": (f"{args.data_dir}/MVBench/json/fine_grained_action.json", f"{args.data_dir}/MVBench/video/Moments_in_Time_Raw/videos/", "video", False),
        "Unexpected Action": (f"{args.data_dir}/MVBench/json/unexpected_action.json", f"{args.data_dir}/MVBench/video/FunQA_test/test/", "video", False),
        "Object Existence": (f"{args.data_dir}/MVBench/json/object_existence.json", f"{args.data_dir}/MVBench/video/clevrer/video_validation/", "video", False),
        "Object Interaction": (f"{args.data_dir}/MVBench/json/object_interaction.json", f"{args.data_dir}/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
        "Object Shuffle": (f"{args.data_dir}/MVBench/json/object_shuffle.json", f"{args.data_dir}/MVBench/video/perception/videos/", "video", False),
        "Character Order": (f"{args.data_dir}/MVBench/json/character_order.json", f"{args.data_dir}/MVBench/video/perception/videos/", "video", False),
        "Egocentric Navigation": (f"{args.data_dir}/MVBench/json/egocentric_navigation.json", f"{args.data_dir}/MVBench/video/vlnqa/", "video", False),
        "Episodic Reasoning": (f"{args.data_dir}/MVBench/json/episodic_reasoning.json", f"{args.data_dir}/MVBench/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
        "Counterfactual Inference": (f"{args.data_dir}/MVBench/json/counterfactual_inference.json", f"{args.data_dir}/MVBench/video/clevrer/video_validation/", "video", False),

        "Anetc Grounding Frame": (f"{args.data_dir}/test-anno/anetc_grounding-frame.json", f"{args.data_dir}/videos/activitynet/", "frame", False),
        "Charades-STA Grounding Frame": (f"{args.data_dir}/test-anno/charades_sta_grounding-frame.json", f"{args.data_dir}/videos/charades/", "video", False),

        "NExTGQA": (f"{args.data_dir}/test-anno/nextgqa.json", f"{args.data_dir}/videos/nextqa/", "video", False),

        "NExTQA": (f"{args.data_dir}/test-anno/nextqa-test.json", f"{args.data_dir}/videos/nextqa/", "video", False),
        "TVQA": (f"{args.data_dir}/test-anno/tvqa-test.json", f"{args.data_dir}/videos/tvqa/", "frame", True),
        "STAR": (f"{args.data_dir}/test-anno/star-test.json", f"{args.data_dir}/videos/charades/", "video", False),
    }

    mvbench_tasks = [
        "Action Sequence", "Action Prediction", "Action Antonym", "Fine-grained Action", "Unexpected Action", "Object Existence", "Object Interaction", "Object Shuffle", "Moving Direction", "Action Localization",
        "Scene Transition", "Action Count", "Moving Count", "Moving Attribute", "State Change", "Fine-grained Pose", "Character Order", "Egocentric Navigation", "Episodic Reasoning", "Counterfactual Inference"
    ]
    if 'MVBench' in args.tasks:
        args.tasks += mvbench_tasks

    data_list = {k: v for k, v in data_list.items() if k in args.tasks}

    #  position embedding
    resolution = 224

    dataset_cls = MVBench_dataset if 'choice' in args.mode else GroundingDataset
    dataset = dataset_cls('', data_list, num_segments=args.num_frames, resolution=resolution, return_msg=args.return_video_msg)

    # load stage2 model
    config_file = args.config
    cfg = Config.from_file(config_file)
    cfg.model.vision_encoder.num_frames = args.num_frames

    use_lora = cfg.model.use_lora
    cfg.model.use_lora = False      # do not add lora in __init__, we will add it later
    cfg.model.num_frame_tokens = args.num_frame_tokens
    model = HawkEye_it(config=cfg.model)
    model.set_device_ids([cfg.device])
    model = model.eval()

    # add lora to run stage3 model
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, 
        r=16, lora_alpha=32, lora_dropout=0.
    )

    if use_lora:
        model.llama_model = get_peft_model(model.llama_model, peft_config)
        model.use_lora = True

    print("loading ckpt from %s" % args.ckpt)
    state_dict = torch.load(args.ckpt, "cpu")
    if 'model' in state_dict.keys():
        msg = model.load_state_dict(state_dict['model'], strict=False)
    else:
        msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model = model.eval()

    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    f_out = open(f"{args.save_path}", "w")

    if args.end_idx is None:
        args.end_idx = len(dataset)

    for example_i in trange(args.start_idx, args.end_idx):
        example = dataset[example_i]
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1

        if args.no_prompt:
            system_prompt = ""
        else:
            if args.system_prompt is not None:
                system_prompt = args.system_prompt
            elif args.mode == 'choice':
                system_prompt = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
            elif args.mode == 'grounding_choice':
                system_prompt = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question, and reply a part of the video that supports your choice.\n"
            else:
                system_prompt = "Assess the displayed video and answer the subsequent question with accuracy.\n"

        if args.no_prompt:
            question_prompt = ""
        else:
            if args.question_prompt is not None:
                question_prompt = args.question_prompt
            elif args.mode in ['choice']:
                question_prompt = "\nOnly give the best option."
            elif args.mode in ['grounding_choice']:
                question_prompt = "\nGive the best option and a part of the video that supports your choice."
            else:
                question_prompt = ""

        if args.return_prompt is not None:
            return_prompt = args.return_prompt
        elif args.mode in ["choice", "grounding_choice"]:
            return_prompt = "(" 
        else:
            return_prompt = ""

        if args.answer_prompt is not None:
            answer_prompt = args.answer_prompt
        elif args.mode in ["choice", "grounding_choice"]:
            answer_prompt = "Best option:(" 
        else:
            answer_prompt = ""

        with torch.no_grad():
            pred = infer_mvbench(
                example, 
                system=system_prompt,
                question_prompt=question_prompt,
                answer_prompt=answer_prompt,
                return_prompt=return_prompt,
                system_q=False,
                print_res=args.print_res or example_i < 20,
                system_llm=True,
                no_prompt=args.no_prompt,
                no_video=args.no_video
            )

        gt_answer = example['answer']
        gt_span = example['span']
        res = check_ans(pred=pred, gt_answer=gt_answer, gt_span=gt_span, video_secs=example['video_secs'], num_frames=args.num_frames, mode=args.mode)
        res_list.append(res)
        f_out.write(json.dumps({'task_type': task_type, 'video_fname': example['video_fname'], 'video_secs': example['video_secs'], 'pred': pred, 'gt_answer': gt_answer, 'gt_span': gt_span, 'res': res}) + '\n')

        if example_i % 20 == 0:
            f_out.flush()

    res_list = [i for i in res_list if i is not None]
    print('num_examples %d, average res %.4lf' % (len(res_list), np.mean(res)))

        # res_list.append({
        #     'pred': pred,
        #     'gt': gt
        # })

        # if check_ans(pred=pred, gt=gt):
        #     acc_dict[task_type][0] += 1
        #     correct += 1
        # print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        # print(f"Total Acc: {correct / total * 100 :.2f}%")
        # print('-' * 30, task_type, '-' * 30)

        # json.dump({
        #     "acc_dict": acc_dict,
        #     "res_list": res_list
        # }, f)
