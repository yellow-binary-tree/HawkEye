import os
import json
import random
import argparse


def load_data(fname):
    data_list = list()
    for line in open(fname):
        info, sent = line.split('##')
        vid, start, end = info.split(' ')
        start, end = float(start), float(end)
        data_list.append({'video': vid + '.mp4', 'start': start, 'end': end, 'caption': sent.strip()})
    return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default='test_grounding')
    parser.add_argument('--video-lengths-fname', type=str, default='data/Charades-STA/video_lengths.json')
    parser.add_argument('--annotation-fname', type=str, default='data/Charades-STA/charades_sta_train.txt')
    parser.add_argument('--instruction-fname', type=str, default='data/VideoChat2-IT/temporal/charades_sta_grounding-choice/instructions.json')
    parser.add_argument('--question', type=str, default='data/VideoChat2-IT/temporal/charades_sta_grounding-choice/questions.json')
    parser.add_argument('--output-fname', type=str, default='data/VideoChat2-IT/video/temporal/charades_sta_grounding-choice/train.json')
    parser.add_argument('--time-span-sent', type=str, default='From frame %d to frame %d.')
    args = parser.parse_args()
    print(args)

    anno = load_data(args.annotation_fname)
    if os.path.exists(args.question):
        args.question = json.load(open(args.question))
    else:
        args.question = [args.question]

    if args.func in ['grounding', 'caption', 'choice']:
        video_lengths = json.load(open(args.video_lengths_fname))
        instructions = json.load(open(args.instruction_fname))
        res = list()
        for example in anno:
            sent = example['caption'].replace('.', '')
            new_example = {'i': random.choice(instructions), 'start_sec': example['start'], 'end_sec': example['end'], 'neg_start_sec': 0, 'neg_end_sec': video_lengths[example['video'][:5]]}
            if args.func == 'grounding':
                new_example['q'] = random.choice(args.question) % sent
                new_example['a'] = args.time_span_sent
            elif args.func == 'caption':
                new_example['a'] = sent
                new_example['q'] = args.time_span_sent
            elif args.func == 'choice':
                options = ["In the middle of the video.", "At the end of the video.", "Throughout the entire video.", "At the beginning of the video."]
                random.shuffle(options)
                options = ["\n(%s) %s" % ("ABCD"[i], opt) for i, opt in enumerate(options)]
                new_example['q'] = "Question: " + random.choice(args.question) % sent + "\nOptions:" + "".join(options)
                new_example['a'] = [([i for i in ["middle", "end", "throughout", "beginning"] if i in opt.lower()][0], opt.strip()) for opt in options]
            res.append({'video': example['video'], 'QA': [new_example]})

    if args.func in ["test_grounding"]:
        res = list()
        for example in anno:
            sent = example['caption'].replace('.', '')
            new_example = {'video': example['video'], 'question': random.choice(args.question) % sent, 'answer': "%.1f-%.1f" % (example['start'], example['end'])}
            new_example = {'video': example['video'], 'question':  sent, 'answer': [example['start'], example['end']]}
            res.append(new_example)

    with open(args.output_fname, 'w') as f_out:
        json.dump(res, f_out)
