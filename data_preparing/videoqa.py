# convert many qa datasets to the input format
import os
import random
import argparse
import json
import pandas as pd
import numpy as np



# nextqa, tvqa and star also have a time span label, but we omit them?
def convert_nextqa_test(input_fname, output_fname, video_mapping_fname):
    video_mapping_dict = json.load(open(video_mapping_fname))
    video_mapping_dict = {key: val + '.mp4' for key, val in video_mapping_dict.items()}
    df = pd.read_csv(input_fname)

    res_list = list()
    for line_i, row in df.iterrows():
        video_fname = video_mapping_dict[str(row['video'])]
        res_dict = {'video': video_fname, 'question': row['question'] + '?', 'qid': row['qid'],
                    'candidates': [row['a%d' % i] + '.' for i in range(5)], 'answer': row['a%d' % row['answer']]}
        res_list.append(res_dict)
    json.dump(res_list, open(output_fname, 'w'))


def convert_tvqa(input_fname, output_fname):
    tv_name_to_folder = {
        'The Big Bang Theory': 'bbt_frames', 'Castle': 'castle_frames', 'How I Met You Mother': 'met_frames',
        "Grey's Anatomy": 'grey_frames', 'Friends': 'friends_frames', 'House M.D.': 'house_frames'
    }

    res_list = list()
    for line in open(input_fname):
        row = json.loads(line)
        start_sec, end_sec = row['ts'].split('-')
        res_dict = {'video': os.path.join(tv_name_to_folder[row['show_name']], row['vid_name']),
                    'question': row['q'], 'qid': row['qid'], 'candidates': [row['a%d' % i] for i in range(5)]}

        if not np.isnan(float(start_sec)) and not np.isnan(float(end_sec)):
            res_dict['start'], res_dict['end'] = float(start_sec), float(end_sec)

        if 'answer_idx' in row:
            res_dict['answer'] = row['a%d' % row['answer_idx']]
        res_list.append(res_dict)
    json.dump(res_list, open(output_fname, 'w'))


def convert_star(input_fname, output_fname):
    res_list = list()
    for row in json.load(open(input_fname)):
        res_dict = {'video': row['video_id'] + '.mp4',
                    'question': row['question'], 'qid': row['question_id'],
                    'candidates': [c['choice'] for c in row['choices']]}
        if 'answer' in row:
            res_dict['answer'] = row['answer']
        res_list.append(res_dict)
    json.dump(res_list, open(output_fname, 'w'))


def convert_star_output(input_fname, output_fname, src_fname):
    '''
    convert videochat2 output on star dataset to the submission format
    '''
    res_dict = {key: [] for key in ['Interaction', 'Sequence', 'Prediction', 'Feasibility']}
    src_data = json.load(open(src_fname))
    pred_data = [json.loads(line) for line in open(input_fname)]
    assert len(src_data) == len(pred_data)

    for example, src_example in zip(pred_data, src_data):
        for key in res_dict.keys():
            if src_example['qid'].startswith(key):
                if example['pred'][1] in 'ABCD':
                    res_dict[key].append({'question_id': src_example['qid'], 'answer': 'ABCD'.index(example['pred'][1])})
                else:
                    print('no choice letter found!')
                    res_dict[key].append({'question_id': src_example['qid'], 'answer': random.choice([0, 1, 2, 3])})
    json.dump(res_dict, open(output_fname, 'w'))


def convert_tvqa_output(input_fname, output_folder, test_src_fname, val_src_fname):
    res_dict = dict()
    src_data = [json.loads(line) for line in open(test_src_fname)]
    pred_data = [json.loads(line) for line in open(input_fname)]
    assert len(src_data) == len(pred_data)

    os.makedirs(output_folder, exist_ok=True)
    for example, src_example in zip(pred_data, src_data):
        pred_id = 'ABCDE'.index(example['pred'][1]) if example['pred'][1] in 'ABCDE' else random.randint(0, 4)
        res_dict[src_example['qid']] = pred_id
    json.dump(res_dict, open(os.path.join(output_folder, 'prediction_test_public.json'), 'w'))

    # generate a random result for the val set
    src_data = [json.loads(line) for line in open(val_src_fname)]
    res_dict = dict()
    for src_example in src_data:
        pred_id = random.randint(0, 4)
        res_dict[src_example['qid']] = pred_id
    json.dump(res_dict, open(os.path.join(output_folder, 'prediction_val.json'), 'w'))

    # generatethe metadata
    metadata = {
        'model_name': '/'.join(input_fname.split('/')[-2:]), 'is_ensemble': False, 'with_ts': True, 'show_on_leaderboard': False,
        'author': 'null', 'institution': 'null', 'description': 'null', 'paper_link': 'null', 'code_link': 'null'
    }
    json.dump(metadata, open(os.path.join(output_folder, 'meta.json'), 'w'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--input2', type=str)
    parser.add_argument('--input3', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    print(args)

    if args.func == 'nextqa-test':
        convert_nextqa_test(args.input, args.output, args.input2)

    if args.func == 'tvqa':
        convert_tvqa(args.input, args.output)

    if args.func == 'star':
        convert_star(args.input, args.output)

    if args.func == 'star_output':
        convert_star_output(args.input, args.output, args.input2)

    if args.func == 'tvqa_output':
        convert_tvqa_output(args.input, args.output, args.input2, args.input3)

