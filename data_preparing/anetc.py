import os
import json
import random
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default='test_grounding')
    parser.add_argument('--annotation-fname', type=str, default='data/ActivityNet/captions/train.json')
    parser.add_argument('--annotation-fname2', type=str, default=None)
    parser.add_argument('--instruction-fname', type=str, default='data/VideoChat2-IT/video/temporal/anetc_grounding-choice/instructions.json')
    parser.add_argument('--question', type=str, default='data/VideoChat2-IT/video/temporal/anetc_grounding-choice/questions.json')
    parser.add_argument('--output-fname', type=str, default='data/VideoChat2-IT/video/temporal/anetc_grounding-choice/train.json')
    parser.add_argument('--time-span-sent', type=str, default='From second %.1f to second %.1f.')       # for frame-level and second-level
    parser.add_argument('--sample-ratio', type=float, default=1.0, help='random sample ratio for the test set')
    args = parser.parse_args()
    print(args)

    os.makedirs(os.path.dirname(args.output_fname), exist_ok=True)
    anno = json.load(open(args.annotation_fname))
    if args.annotation_fname2 is not None:
        anno1, anno = anno, dict()
        anno2 = json.load(open(args.annotation_fname2))
        keys = anno1.keys() | anno2.keys()
        for key in keys:
            if key in anno1:
                anno[key] = anno1[key]
                if key in anno2:
                    anno[key]['sentences'] = anno1[key]['sentences'] + anno2[key]['sentences']
                    anno[key]['timestamps'] = anno1[key]['timestamps'] + anno2[key]['timestamps']
            else:
                anno[key] = anno2[key]

    if os.path.exists(args.question):
        args.question = json.load(open(args.question))
    else:
        args.question = [args.question]

    if args.func in ['grounding', 'caption', 'choice']:
        instructions = json.load(open(args.instruction_fname))
        res = list()
        for video_id, video_data in anno.items():
            for (start_sec, end_sec), sent in zip(video_data['timestamps'], video_data['sentences']):
                if sent.endswith('.'):
                    sent = sent[:-1]
                new_example = {'i': random.choice(instructions), 'start_sec': start_sec, 'end_sec': end_sec, 'neg_start_sec': 0, 'neg_end_sec': video_data['duration']}
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
                res.append({'video': video_id, 'QA': [new_example]})

    if args.func in ["test_grounding"]:
        res = list()
        for video_id, video_data in anno.items():
            for (start_sec, end_sec), sent in zip(video_data['timestamps'], video_data['sentences']):
                if random.random() > args.sample_ratio: continue
                sent = sent.replace('.', '').strip()
                new_example = {'video': video_id, 'question': sent, "answer": [start_sec, end_sec]}
                res.append(new_example)

    print('saved %d examples' % len(res))
    with open(args.output_fname, 'w') as f_out:
        json.dump(res, f_out)
