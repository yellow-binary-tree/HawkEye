import os as __os  # add "__" if not want to be exported

anno_root_it = "data/HawkEye-IT"

# ============== pretraining datasets=================

available_corpus = dict(
    caption_textvr={
        "dataset_name": "caption_textvr",
        "label_file": f"{anno_root_it}/video/caption/textvr/train.json", 
        "data_root": "data/videos/textvr",
        "media_type": "video",
    },

    caption_videochat={
        "dataset_name": "caption_videochat",
        "label_file": f"{anno_root_it}/video/caption/videochat/train.json",
        "data_root": "data/videos/webvid",
        "media_type": "video",
    },

    caption_webvid={
        "dataset_name": "caption_webvid",
        "label_file": f"{anno_root_it}/video/caption/webvid/train.json",
        "data_root": "data/videos/webvid",
        "media_type": "video",
    },

    caption_youcook2={
        "dataset_name": "caption_youcook2",
        "label_file": f"{anno_root_it}/video/caption/youcook2/train.json", 
        "data_root": "data/videos/youcook2",
        "media_type": "video",
    },

    classification_k710={
        "dataset_name": "classification_k710",
        "label_file": f"{anno_root_it}/video/classification/k710/train.json",
        "data_root": "data/videos/kinetics",
        "media_type": "video",
    },

    classification_ssv2={
        "dataset_name": "classification_ssv2",
        "label_file": f"{anno_root_it}/video/classification/ssv2/train.json",
        "data_root": "data/videos/ssv2",
        "media_type": "video",
    },

    conversation_videochat1={
        "dataset_name": "conversation_videochat1",
        "label_file": f"{anno_root_it}/video/conversation/videochat1/train.json",
        "data_root": "data/videos/webvid",
        "media_type": "video",
    },

    conversation_videochatgpt={
        "dataset_name": "conversation_videochatgpt",
        "label_file": f"{anno_root_it}/video/conversation/videochatgpt/train.json",
        "data_root": "data/videos/activitynet",
        "media_type": "video",
    },

    reasoning_next_qa={
        "dataset_name": "reasoning_next_qa",
        "label_file": f"{anno_root_it}/video/reasoning/next_qa/train.json",
        "data_root": "data/videos/nextqa",
        "media_type": "video",
    },

    reasoning_clevrer_qa={
        "dataset_name": "reasoning_clevrer_qa",
        "label_file": f"{anno_root_it}/video/reasoning/clevrer_qa/train.json",
        "data_root": "data/videos/clevrer",
        "media_type": "video",
    },

    reasoning_clevrer_mc={
        "dataset_name": "reasoning_clevrer_mc",
        "label_file": f"{anno_root_it}/video/reasoning/clevrer_mc/train.json",
        "data_root": "data/videos/clevrer",
        "media_type": "video",
    },

    vqa_tgif_frame_qa={
        "dataset_name": "vqa_tgif_frame_qa",
        "label_file": f"{anno_root_it}/video/vqa/tgif_frame_qa/train.json",
        "data_root": "data/videos/tgif",
        "media_type": "video",
    },

    vqa_tgif_transition_qa={
        "dataset_name": "vqa_tgif_transition_qa",
        "label_file": f"{anno_root_it}/video/vqa/tgif_transition_qa/train.json",
        "data_root": "data/videos/tgif",
        "media_type": "video",
    },

    vqa_webvid_qa={
        "dataset_name": "vqa_webvid_qa",
        "label_file": f"{anno_root_it}/video/vqa/webvid_qa/train.json",
        "data_root": "data/videos/webvid",
        "media_type": "video",
    },

    internvid_grounding={
        "dataset_name": "internvid_grounding", 'grounding_method': 'choice',
        "label_file": f"{anno_root_it}/video/temporal/internvid_grounding/train.json",
        "data_root": "data/videos/internvid-g",
        "media_type": "video",
    },

    internvid_caption={
        "dataset_name": "internvid_grounding", 'grounding_method': 'choice',
        "label_file": f"{anno_root_it}/video/temporal/internvid_caption/train.json",
        "data_root": "data/videos/internvid-g",
        "media_type": "video",
    },

)

# select the instruction training data you have
available_corpus["hawkeye_instruction"] = [
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_webvid"],
    available_corpus["caption_youcook2"],
    available_corpus["classification_k710"],
    available_corpus["classification_ssv2"],
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"],
    available_corpus["reasoning_clevrer_mc"],
    available_corpus["vqa_ego_qa"],
    available_corpus["vqa_tgif_frame_qa"],
    available_corpus["vqa_tgif_transition_qa"],
    available_corpus["vqa_webvid_qa"],
    available_corpus["internvid_grounding"],
    available_corpus["internvid_caption"],
]
