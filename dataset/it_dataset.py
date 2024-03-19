import logging
import os
import json
import copy
import random
from os.path import basename

import numpy as np

from dataset.base_dataset import ImageVideoBaseDataset
from dataset.utils import load_anno
from dataset.video_utils import VIDEO_READER_FUNCS
from utils.distributed import is_main_process
from configs.dataset_utils import VIDEO_PATH_MAPPING

logger = logging.getLogger(__name__)


class ITImgTrainDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(
        self, dataset_name, label_file, data_root, media_type, transform, 
        system="", role=("Human", "Assistant"),
        start_token="<Image>", end_token="</Image>",
        random_shuffle=True, # if True, shuffle the QA list
    ):
        super().__init__()
        self.dataset_name, self.label_file, self.data_root, self.media_type = dataset_name, label_file, data_root, media_type

        logger.info('Load json file')
        with open(self.label_file, 'r') as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)
        self.transform = transform

        # prompt parameters
        if system:
            assert system[-1] == " ", "' ' should be add in the end of system, thus '###' will be tokenized into one token."
        # currently not support add start_token and end_token in the system, since the msg should be added properly
        self.begin_signal = "###"
        self.end_signal = " "
        self.start_token = start_token
        self.end_token = end_token
        self.system = system
        self.role = role
        self.random_shuffle = random_shuffle
        # instruction location and number
        logger.info(f"Random shuffle: {self.random_shuffle}")

    def get_anno(self, index):
        filename = self.anno[index][self.media_type]
        qa = self.anno[index]["QA"]
        if "start" in self.anno[index] and "end" in self.anno[index]:
            anno = {
                "image": os.path.join(self.data_root, filename), "qa": qa,
                "start": self.anno[index]["start"], "end": self.anno[index]["end"],
            }
        else:
            anno = {"image": os.path.join(self.data_root, filename), "qa": qa}
        return anno

    def __len__(self):
        return self.num_examples
    
    def process_qa(self, qa, msg=""):
        cur_instruction = ""
        # randomly shuffle qa for conversation
        if self.random_shuffle and len(qa) > 1:
            random.shuffle(qa)
        if "i" in qa[0].keys() and qa[0]["i"] != "":
            cur_instruction = qa[0]["i"] + self.end_signal

        conversation = self.system
        # add instruction as system message
        if cur_instruction:
            conversation += cur_instruction

        # rstrip() for the extra " " in msg
        conversation += (
            self.begin_signal + self.role[0] + ": " + 
            self.start_token + self.end_token + msg.rstrip() + self.end_signal
        )

        for sentence in qa:
            q = sentence["q"]
            a = sentence["a"]
            if q != "":
                conversation += (self.begin_signal + self.role[0] + ": " + q + self.end_signal)
            else:
                # no question, often in caption dataset
                pass
            conversation += (self.begin_signal + self.role[1] + ": " + a + self.end_signal)
        conversation += self.begin_signal
        
        if cur_instruction:
            cur_instruction += qa[0]["q"]
        return conversation, cur_instruction.strip()

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            image, index = self.load_and_transform_media_data_image(index, ann["image"])
            conversation, instruction = self.process_qa(ann["qa"])
            return image, conversation, instruction, index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading image {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class ITVidTrainDataset(ITImgTrainDataset):
    def __init__(self, dataset_name, label_file, data_root, media_type, transform,
        num_frames=4, video_reader_type="decord", sample_type="rand", num_tries=3,
        system="", role=("Human", "Assistant"),
        start_token="<Video>", end_token="</Video>",
        add_second_msg=True, random_shuffle=True, grounding_method=None, random_video_truncate=True,
        min_gold_clips=1, max_gold_clips=None, fps=None, num_examples=None):

        super().__init__(
            dataset_name, label_file, data_root, media_type, transform, system, role, start_token, end_token, random_shuffle
        )

        self.num_frames, self.video_reader_type, self.sample_type, self.num_tries, self.fps = num_frames, video_reader_type, sample_type, num_tries, fps
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        
        self.add_second_msg = add_second_msg
        if add_second_msg:
            logger.info(f"Add second message: The video contains X frames sampled at T seconds.")

        # video data path mapping, for training on different devices that videos are stored in different places
        self.video_path_mapping = VIDEO_PATH_MAPPING.get(self.dataset_name, lambda x: x)
        new_anno = list()
        for example in self.anno:
            new_video_fname = self.video_path_mapping(example[self.media_type])
            if new_video_fname is not None:
                example[self.media_type] = new_video_fname
                new_anno.append(example)
        self.old_anno = self.anno
        self.anno = new_anno
        if num_examples is not None and num_examples < len(self.anno):
            logger.info("random sampling from %d examples to %d examples" % (len(self.anno), num_examples))
            self.anno = random.sample(self.anno, num_examples)

        self.num_examples = len(self.anno)

        # check if the video is a folder of frames
        if os.path.isdir(os.path.join(self.data_root, self.anno[0][self.media_type])):
            self.video_reader_type = "frames"
            self.video_reader = VIDEO_READER_FUNCS[self.video_reader_type]
            assert hasattr(self, "fps"), "fps must be specified for folder of frames"

        # for grounding dataset
        self.random_video_truncate = random_video_truncate
        self.grounding_method = grounding_method        # if None, then this dataset is a normal video dataset, not a grounding dataset.
        self.min_gold_clips = min_gold_clips
        self.max_gold_clips = max_gold_clips if max_gold_clips is not None else num_frames // 2

        logger.info('%s dataset with length %d' % (self.dataset_name, len(self)))
        logger.info('example data: {}'.format(self.get_example_data()))

    def get_example_data(self):
        video, conversation, instruction, index = self[0]
        return video.size(), conversation, instruction, index

    def __getitem__(self, index):
        try:
            # my grounding dataset and the video dataset in video_chat2 have the same data format, they only differs at the processing the data.
            # so we need to use the same dataset for them.

            if self.grounding_method is None:       # not a grounding dataset
                ann = self.get_anno(index)
                msg = ""
                clip = None
                if "start" in ann and "end" in ann:
                    clip = [ann["start"], ann["end"]]

                video, index, sec = self.load_and_transform_media_data_video(index, ann["image"], return_fps=True, clip=clip)
                if self.add_second_msg:
                    # " " should be added in the start and end
                    msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. "
                conversation, instruction = self.process_qa(ann["qa"], msg)
                if video.size(0) != self.num_frames:
                    raise ValueError(f"Error num frames {video.size()}")
                return video, conversation, instruction, index

            else:
                ann = self.get_anno(index)
                msg = ""

                if self.grounding_method == 'direct_caption':
                    video_start_sec, video_end_sec = ann["qa"][0]['start_sec'], ann["qa"][0]['end_sec']
                else:
                    (video_start_sec, video_end_sec), (gt_start, gt_end), choice = self.get_input_video_span(ann["qa"][0])
                video, index, sec = self.load_and_transform_media_data_video(
                    index, ann["image"], return_fps=True, clip=[video_start_sec, video_end_sec])

                if self.add_second_msg:
                    # " " should be added in the start and end
                    msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. "

                qa_data = copy.deepcopy(ann['qa'])

                if self.grounding_method == "time_caption":
                    caption = qa_data[0]['q'].replace('.', '').lower()
                    time_information = {
                        "middle": "in the middle of the video", "end": "at the end of the video",
                        "throughout": "throughout the entire video", "beginning": "at the beginning of the video"
                    }[choice]
                    if random.random() < 0.5:       # add time information before caption
                        caption = self.start_token + self.end_token + time_information + ' ' + caption
                    else:       # add time information after caption
                        caption = self.start_token + self.end_token + caption + ' ' + time_information
                    return video, caption, '', index

                elif self.grounding_method == 'direct_caption':
                    caption = self.start_token + self.end_token + qa_data[0]['q'].lower()
                    return video, caption, '', index

                elif self.grounding_method == "choice":
                    if isinstance(qa_data[0]['a'], list):       # this is a grounding task, where the answer is a choice
                        qa_data[0]['a'] = [i[1] for i in qa_data[0]['a'] if i[0] == choice][0]
                    elif isinstance(qa_data[0]['q'], list):     # this is a captioning task, where the question is a choice
                        qa_data[0]['q'] = [i[1] for i in qa_data[0]['q'] if i[0] == choice][0]

                else:
                    if "%" in qa_data[0]['q']:
                        qa_data[0]['q'] = qa_data[0]['q'] % (gt_start, gt_end)
                    if "%" in qa_data[0]['a']:
                        qa_data[0]['a'] = qa_data[0]['a'] % (gt_start, gt_end)

                conversation, instruction = self.process_qa(qa_data, msg)
                return video, conversation, instruction, index

        except Exception as e:
            # raise e
            logger.warning(f"Caught exception {e} when loading video {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)

    def get_input_video_span(self, example):
        if not self.random_video_truncate:
            video_start_sec, video_end_sec = example['neg_start_sec'], example['neg_end_sec']
            if self.grounding_method == "second":
                return (video_start_sec, video_end_sec), (example['start_sec'] - video_start_sec), (example['end_sec'] - video_start_sec)

            if self.grounding_method in ["frame", "frame_token"]:
                video_length_sec = video_end_sec - video_start_sec
                start_pos, end_pos = (example['start_sec'] - video_start_sec) / video_length_sec, (example['end_sec'] - video_start_sec) / video_length_sec
                start_pos, end_pos = min(max(start_pos, 0), 1), min(max(end_pos, 0), 1)
                start_frame, end_frame = min(self.num_frames - 1, np.floor(start_pos * self.num_frames)), min(self.num_frames - 1, np.floor(end_pos * self.num_frames))
                return (video_start_sec, video_end_sec), (start_frame, end_frame), None

            if self.grounding_method in ["choice", "time_caption"]:
                if example['end_sec'] < (video_end_sec + video_start_sec) / 2:
                    return (video_start_sec, video_end_sec), (None, None), 'beginning'
                if example['start_sec'] > (video_end_sec + video_start_sec) / 2:
                    return (video_start_sec, video_end_sec), (None, None), 'end'
                if example['end_sec'] - example['start_sec'] < (video_end_sec - video_start_sec) / 2:
                    return (video_start_sec, video_end_sec), (None, None), 'middle'
                return (video_start_sec, video_end_sec), (None, None), 'throughout'

        min_caption_ratio = self.min_gold_clips / self.num_frames      # if the target segment is too short, a frame of it will even not be sampled.
        max_caption_ratio = self.max_gold_clips / self.num_frames      # for second / frame format, if the segment is too long, the training effect on discriminating target segments will be diminished
        if self.grounding_method in ["choice", "time_caption"]:     # caption task, or coarse-grained rep. grounding task formatted as choices
            # check what choices are available
            caption_clip_length_sec = example['end_sec'] - example['start_sec']
            available_choices = [i[0] for i in [
                ('beginning', example['neg_end_sec'] - example['end_sec'] > caption_clip_length_sec),
                ('end', example['start_sec'] - example['neg_start_sec'] > caption_clip_length_sec),
                ('middle', example['start_sec'] - example['neg_start_sec'] > caption_clip_length_sec / 2 and example['neg_end_sec'] - example['end_sec'] > caption_clip_length_sec / 2),
                ('throughout', True),
            ] if i[1]]
            weights = [0.3 if c == 'throughout' else 2 for c in available_choices]      # this weight is set to make the 

            for _ in range(10):
                choice = random.choices(available_choices, weights=weights, k=1)[0]
                # sample start sec and end sec of the video according to the choice
                if choice == 'beginning':
                    video_end_sec = random.uniform(example['end_sec'] + caption_clip_length_sec, example['neg_end_sec'])
                    video_start_sec = random.uniform(max(example['neg_start_sec'], 2 * example['end_sec'] - video_end_sec), example['start_sec'])
                elif choice == 'middle':
                    mid_sec = random.uniform(max(example['start_sec'], example['neg_start_sec'] * 0.5 + example['end_sec'] * 0.75 - example['start_sec'] * 0.25),
                                            min(example['end_sec'], example['neg_end_sec'] * 0.5 + example['start_sec'] * 0.75 - example['end_sec'] * 0.25))
                    half_length = random.uniform(caption_clip_length_sec / 2 + max(mid_sec - example['start_sec'], example['end_sec'] - mid_sec),
                                                min(mid_sec - example['neg_start_sec'], example['neg_end_sec'] - mid_sec))
                    video_start_sec = mid_sec - half_length
                    video_end_sec = mid_sec + half_length
                elif choice == 'end':
                    video_start_sec = random.uniform(example['neg_start_sec'], example['start_sec'] - caption_clip_length_sec)
                    video_end_sec = random.uniform(example['end_sec'], min(example['neg_end_sec'], 2 * example['start_sec'] - video_start_sec))
                elif choice == 'throughout':
                    video_start_sec = random.uniform(max(example['neg_start_sec'], (3 * example['start_sec'] - example['end_sec']) / 2), example['start_sec'])
                    video_end_sec = random.uniform(example['end_sec'], min(example['neg_end_sec'], (3 * example['end_sec'] - example['start_sec']) / 2))
                gt_start_sec, gt_end_sec = (example['start_sec'] - video_start_sec), (example['end_sec'] - video_start_sec)

                # the logic of the sampling starting point and ending point above is confusing enough. If the constraints of min_caption_ratio are added, it will only make the difference of these codes worse.
                # Therefore, this requirement can only be met through the method of "return and redo if it does not meet"
                if (video_end_sec - video_start_sec) * min_caption_ratio < gt_end_sec - gt_start_sec:
                    break
            return (video_start_sec, video_end_sec), (gt_start_sec, gt_end_sec), choice

        # for second/frame format grounding
        # 1. sample start and end secs according to min_caption_ratio and max_caption_ratio
        caption_clip_length_sec = example['end_sec'] - example['start_sec']
        max_video_length_sec = min(caption_clip_length_sec / min_caption_ratio, example['neg_end_sec'] - example['neg_start_sec'])
        min_video_length_sec = min(caption_clip_length_sec / max_caption_ratio, example['neg_end_sec'] - example['neg_start_sec'])

        video_length_sec = random.uniform(min_video_length_sec, max_video_length_sec)
        video_mid_sec_interval = [
            max(example['end_sec'] - video_length_sec / 2, example['neg_start_sec'] + video_length_sec / 2),
            min(example['start_sec'] + video_length_sec / 2, example['neg_end_sec'] - video_length_sec / 2),
        ]
        video_mid_sec = random.uniform(video_mid_sec_interval[0], video_mid_sec_interval[1])
        video_start_sec, video_end_sec = video_mid_sec - video_length_sec / 2, video_mid_sec + video_length_sec / 2

        # 2. get position of pos span in the video
        if self.grounding_method == "second":
            gt_start_sec, gt_end_sec = (example['start_sec'] - video_start_sec), (example['end_sec'] - video_start_sec)
            return (video_start_sec, video_end_sec), (gt_start_sec, gt_end_sec), None

        if self.grounding_method in ['frame', "frame_token"]:
            start_pos, end_pos = (example['start_sec'] - video_start_sec) / video_length_sec, (example['end_sec'] - video_start_sec) / video_length_sec
            start_pos, end_pos = min(max(start_pos, 0), 1), min(max(end_pos, 0), 1)
            start_frame, end_frame = min(self.num_frames - 1, np.floor(start_pos * self.num_frames)), min(self.num_frames - 1, np.floor(end_pos * self.num_frames))
            return (video_start_sec, video_end_sec), (start_frame, end_frame), None
