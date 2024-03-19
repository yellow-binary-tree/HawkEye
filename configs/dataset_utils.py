# some logit for loading data

import os
import logging
logger = logging.getLogger(__name__)


class WebvidPathMapping:
    def __init__(self, video_ids_fname) -> None:
        self.video_ids = set()
        self.video_ids_fname = video_ids_fname

    def __call__(self, input_fname):
        if not self.video_ids:
            self.video_ids = set([line.strip() for line in open(self.video_ids_fname)])
            logger.info("In WebvidPathMapping, there are %d available videos in folder %s" % (len(self.video_ids), self.video_ids_fname))
        fname = input_fname.split("/")[-1].split('.')[0]
        if fname not in self.video_ids:
            return None
        return os.path.join(fname[:3], fname)


def anet_path_mapping(input_fname):
    fname = input_fname.split("/")[-1].split('.')[0]
    return fname


def clevrer_path_mapping(input_fname):
    '''
    video_02238.mp4 to video_02000-03000/video_02238.mp4
    '''
    interval = int(input_fname.split('.')[0].split('_')[-1]) // 1000
    folder = 'video_%05d-%05d' % (interval * 1000, interval * 1000 + 1000)
    return os.path.join(folder, input_fname)


webvid_path_mapping = WebvidPathMapping('data/WebVid/video_ids.txt')

VIDEO_PATH_MAPPING = {
    'caption_webvid': webvid_path_mapping,
    'caption_videochat': webvid_path_mapping,
    'conversation_videochat1': webvid_path_mapping,
    'vqa_webvid_qa': webvid_path_mapping,
    'conversation_videochatgpt': anet_path_mapping,
    'reasoning_clevrer_qa': clevrer_path_mapping,
    'reasoning_clevrer_mc': clevrer_path_mapping,
}