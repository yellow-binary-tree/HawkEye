folder=$1
ckpt=$2
max_turns=4

# test on charades-sta
python tasks/test_recursive_grounding.py --max-turns ${max_turns} --config ${folder}/config.json \
    --ckpt ${folder}/ckpt_${ckpt}.pth \
    --video-path data/videos/charades \
    --data-path data/test-anno/charades_sta-recursive_grounding.json \
    --save-path ${folder}/charades_sta-recursive_grounding-${ckpt}-${max_turns}_turns.jsonl \
    > ${folder}/charades_sta-recursive_grounding-${ckpt}-${max_turns}_turns.log 2>&1 &
wait

# test on anet-captions
python tasks/test_recursive_grounding.py --max-turns ${max_turns} --config ${folder}/config.json \
    --ckpt ${folder}/ckpt_${ckpt}.pth \
    --video-path data/videos/activitynet \
    --data-path data/test-anno/anetc-recursive_grounding.json \
    --save-path ${folder}/anetc-recursive_grounding-${ckpt}-${max_turns}_turns.jsonl \
    > ${folder}/anetc-recursive_grounding-${ckpt}-${max_turns}_turns.log 2>&1 &
wait

# test on nextgqa
python tasks/test_recursive_grounding.py --max-turns ${max_turns} --config ${folder}/config.json \
    --ckpt ${folder}/ckpt_${ckpt}.pth \
    --video-path data/videos/nextqa \
    --data-path data/test-anno/nextgqa-recursive_grounding.json \
    --save-path ${folder}/nextgqa-recursive_grounding-${ckpt}-${max_turns}_turns.jsonl \
    > ${folder}/nextgqa-recursive_grounding-${ckpt}-${max_turns}_turns.log 2>&1 &
wait
