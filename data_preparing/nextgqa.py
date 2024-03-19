import json
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default='grounding_qa')
    parser.add_argument('--annotation-fname', type=str, default='data/NExTQA/nextgqa/test.csv')
    parser.add_argument('--grounding-fname', type=str, default='data/NExTQA/nextgqa/gsub_test.json')
    parser.add_argument('--pred-span-fname', type=str)
    parser.add_argument('--video-mapping-fname', type=str, default='data/NExTQA/nextgqa/map_vid_vidorID.json')
    parser.add_argument('--output-fname', type=str, default='data/MVBench/json/nextgqa.json')
    args = parser.parse_args()
    print(args)

    video_mapping_dict = json.load(open(args.video_mapping_fname))
    video_mapping_dict = {key: val + '.mp4' for key, val in video_mapping_dict.items()}
    df = pd.read_csv(args.annotation_fname)

    grounding_dict, video_lengths = dict(), dict()
    for video_key, data in json.load(open(args.grounding_fname)).items():
        video_fname = video_mapping_dict[video_key]
        grounding_dict[video_fname] = dict()
        video_lengths[video_fname] = data['duration']
        for question_key, spans in data['location'].items():
            grounding_dict[video_fname][question_key] = spans

    if args.func in ["test_grounding"]:
        res_list = list()
        for line_i, row in df.iterrows():   
            video_fname = video_mapping_dict[str(row['video_id'])]
            res_dict = {'video': video_fname, 'question': row['question'] + '?', 'duration': video_lengths[video_fname], 'qid': row['qid'], 'answer': grounding_dict[video_fname][str(row['qid'])]}
            res_list.append(res_dict)
        json.dump(res_list, open(args.output_fname, 'w'))