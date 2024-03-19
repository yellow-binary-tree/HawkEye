import os
import glob
import json
import random
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default='grounding', choices=['grounding', 'caption', 'choice', 'choice_caption'])
    parser.add_argument('--annotation-fname', type=str, default='data/InternVid-G/train.jsonl')
    parser.add_argument('--instruction-fname', type=str, default='data/VideoChat2-IT/video/temporal/internvid_grounding/instructions.json')
    parser.add_argument('--question', type=str, default='data/VideoChat2-IT/video/temporal/internvid_grounding/questions.json')
    parser.add_argument('--output-fname', type=str, default='data/VideoChat2-IT/video/temporal/internvid_grounding/train.json')
    parser.add_argument('--time-span-sent', type=str, default='From second %.1f to second %.1f.')       # used for frame-level / second-level rep.
    args = parser.parse_args()
    print(args)

    res = list()

    instructions = json.load(open(args.instruction_fname))
    input_fnames = glob.glob(args.annotation_fname)

    if os.path.exists(args.question):
        args.question = json.load(open(args.question))
    else:
        args.question = [args.question]

    for input_fname in input_fnames:
        print('loading data from', input_fname)
        for line in open(input_fname).readlines():
            example = json.loads(line)

            example['caption'] = example['caption'][0].upper() + example['caption'][1:]
            if example['caption'].endswith('.'):
                example['caption'] = example['caption'][:-1]

            new_example = {k: example[k] for k in ['start_sec', 'end_sec', 'neg_start_sec', 'neg_end_sec']}
            new_example['i'] = random.choice(instructions)
            if args.func == 'grounding':
                new_example['q'] = random.choice(args.question) % example['caption']
                new_example['a'] = args.time_span_sent
            elif args.func == 'caption':
                new_example['a'] = example['caption']
                new_example['q'] = args.time_span_sent
            elif args.func == 'choice':
                options = ["In the middle of the video.", "At the end of the video.", "Throughout the entire video.", "At the beginning of the video."]
                random.shuffle(options)
                options = ["\n(%s) %s" % ("ABCD"[i], opt) for i, opt in enumerate(options)]
                new_example['q'] = "Question: " + random.choice(args.question) % example['caption'] + "\nOptions:" + "".join(options)
                new_example['a'] = [([i for i in ["middle", "end", "throughout", "beginning"] if i in opt.lower()][0], opt.strip()) for opt in options]
            elif args.func == 'choice_caption':
                options = ["In the middle of the video.", "At the end of the video.", "Throughout the entire video.", "At the beginning of the video."]
                new_example['q'] = [([i for i in ["middle", "end", "throughout", "beginning"] if i in opt.lower()][0], opt.strip()) for opt in options]
                new_example['a'] = example['caption']
            res.append({'video': example['video'], 'QA': [new_example]})

    with open(args.output_fname, 'w') as f_out:
        json.dump(res, f_out)
