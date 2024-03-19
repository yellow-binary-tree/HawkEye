import sys
sys.path.append('.')

import os
import re
import json
from tqdm import tqdm, trange
import argparse
import random

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
from models.hawkeye2_it import HawkEye_it
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
                img_list=video_list, max_new_tokens=20,
                answer_prompt=answer_prompt, print_res=print_res
            )[0]
    # remove potential explanation
    # llm_message = return_prompt + llm_message.strip().split('\n')[0]
    # sometimes we need this explanation
    llm_message = return_prompt + llm_message
    return llm_message


def check_ans(pred_span, gt_spans):
    return max([calculate_iou(pred_span, gt_span) for gt_span in gt_spans])

def calculate_iou(pred_span, gt_span):
    pred_start, pred_end = pred_span
    gold_start, gold_end = gt_span
    intersection = max(0, min(gold_end, pred_end) - max(gold_start, pred_start))
    union = max(0, max(gold_end, pred_end) - min(gold_start, pred_start))
    if union <= 0 or intersection <= 0:
        return 0
    return intersection / union


class RecursiveGroundingEvaluator:
    def __init__(self, model, max_turns=2, num_segments=12, resolution=224, prompts=[], return_msg=True):
        self.model = model
        self.max_turns = max_turns

        # transform
        self.num_segments = num_segments
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

        self.return_msg = return_msg
        self.system_prompt, self.question_prompt, self.answer_prompt, self.return_prompt = prompts
        self.choice_to_answer = {
            "middle": "In the middle of the video.",
            "end": "At the end of the video.",
            "throughout": "Throughout the entire video.",
            "beginning": "At the beginning of the video."
        }

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
        if os.path.isdir(video_path):       # this is a folder of frames
            return self.read_frame(video_path, bound)

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
            start_sec = 0 if bound is None else bound[0]
            sec = ", ".join([str(round(f / fps - start_sec, 1)) for f in frame_indices])
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


    def inference(self, example, print_res=False):
        '''
        max_turns: the max amount of turns of ank & answer.
        returns: [pred_start_sec, pred_end_sec], turns_used
        '''
        _, _, duration_secs = self.read_video(example['video'])
        bound, pred_answer_list = None, list()
        for turn_i in range(self.max_turns):
            bound, pred_answer = self.single_turn(example['question'], example['video'], bound, print_res=print_res)
            if print_res:
                print(f"Predicted Bound of Turn {turn_i + 1}: {bound}")
            pred_answer_list.append(pred_answer)
            if pred_answer == 'throughout':
                break
        return bound, pred_answer_list, duration_secs

    def single_turn(self, question, video_path, bound, print_res=False):
        '''
        bound: [start_sec, end_sec]
        returns: pred_bound, need_another_turn
        '''
        torch_imgs, video_msg, video_secs = self.read_video(video_path, bound)
        if bound is None:
            bound = [0, video_secs]

        candidates = list(self.choice_to_answer.items())
        random.shuffle(candidates)
        candidates = [(c[0], '(' + 'ABCD'[i] + ') ' + c[1]) for i, c in enumerate(candidates)]
        candidates_text = '\n'.join([c[1] for c in candidates])
        question = question + '\nOptions:\n' + candidates_text
        example = {
            'video': torch_imgs, 
            'question': question, 
            'video_msg': video_msg,
        }
        llm_message = infer_mvbench(
            data_sample=example, system=self.system_prompt, question_prompt=self.question_prompt,
            answer_prompt=self.answer_prompt, return_prompt=self.return_prompt,
            system_llm=True, print_res=print_res,
        )

        pred_answer = 'throughout'
        for answer, reply in candidates:
            if llm_message.lower().strip().startswith(reply.lower().strip()):
                pred_answer = answer
                break

        mid_sec, quarter_length = (bound[1] + bound[0]) / 2, (bound[1] - bound[0]) / 4
        if pred_answer == 'throughout':
            pred_bound = bound
        elif pred_answer == 'beginning':
            pred_bound = [bound[0], mid_sec]
        elif pred_answer == 'end':
            pred_bound = [mid_sec, bound[1]]
        elif pred_answer == 'middle':
            pred_bound = [mid_sec - quarter_length, mid_sec + quarter_length]

        return pred_bound, pred_answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.json")
    parser.add_argument("--ckpt", type=str, default="model/VideoChat2/videochat2_7b_stage3.pth")
    parser.add_argument("--data-path", type=str, default="data/MVBench/json/nextgqa.json")
    parser.add_argument("--video-path", type=str, default="data/MVBench/video/nextqa")
    parser.add_argument("--save-path", type=str, default="test.json")
    parser.add_argument("--print-res", type=int, default=0)
    parser.add_argument("--no-video", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--return-video-msg", type=int, default=1)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--num-frame-tokens", type=int, default=0)
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--question-pattern", type=str, default=None)
    parser.add_argument("--no-prompt", action='store_true')
    args = parser.parse_args()

    if args.no_prompt:
        args.return_video_msg = 0

    print(args)

    # start loading models
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

    if args.no_prompt:
        system_prompt = ""
    else:
        if args.system_prompt is not None:
            system_prompt = args.system_prompt
        else:
            system_prompt = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select a part of video that relates most to the following sentence.\n"
            system_prompt = "Examine the video and choose the most appropriate choice in accordance with the video's content.\n"

    if args.no_prompt:
        question_prompt = ""
    else:
        question_prompt = "\nOnly give the best option."

    return_prompt = "("
    answer_prompt = "Best option: ("

    # load dataset and evaluator
    res_list = []
    examples = json.load(open(args.data_path))
    evaluator = RecursiveGroundingEvaluator(model, max_turns=args.max_turns, num_segments=args.num_frames, prompts=[system_prompt, question_prompt, answer_prompt, return_prompt])
    f_out = open(f"{args.save_path}", "w")
    if args.end_idx is None:
        args.end_idx = len(examples)

    if args.question_pattern is not None:
        question_pattern = args.question_pattern
    else:
        question_pattern = "Question: During which part of the video does '%s' occur?"

    for example_i in trange(args.start_idx, args.end_idx):
        example = examples[example_i]
        question = question_pattern % example['question']
        model_inputs = {'video': os.path.join(args.video_path, example['video']), 'question': question}
        with torch.no_grad():
            try:
                pred_span, pred_answer_list, duration_secs = evaluator.inference(model_inputs, print_res=(example_i - args.start_idx) < 50)
            except Exception as e:
                print('error at question %d, using empty result' % example_i)
                pred_span, pred_answer_list, duration_secs = [0, 0], [None], None

        if 'span' in example:
            gt_span = example['span'] if isinstance(example['span'][0], list) else [example['span']]
            res = check_ans(pred_span, gt_span)
        else:
            gt_span, res = None, 0

        res_list.append(res)
        f_out.write(json.dumps({
            'video_fname': model_inputs['video'], 'duration': duration_secs, 'question': model_inputs['question'],
            'pred_span': pred_span, 'gt_span': gt_span, 'pred_answer_list': pred_answer_list, 'iou': res}) + '\n')

        if example_i % 20 == 0:
            f_out.flush()

    print('num_examples %d, average res %.4lf' % (len(res_list), np.mean(res)))
