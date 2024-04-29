max_turns=4

# test on charades-sta
python tasks/test_recursive_grounding.py --max-turns ${max_turns} --config configs/config.json \
    --ckpt model/hawkeye.pth \
    --video-path data/videos/charades \
    --data-path data/test-anno/charades_sta-recursive_grounding.json \
    --save-path outputs/charades_sta-recursive_grounding-${max_turns}_turns.jsonl \
    > outputs/charades_sta-recursive_grounding-${max_turns}_turns.log 2>&1 &
wait

# test on anet-captions
python tasks/test_recursive_grounding.py --max-turns ${max_turns} --config configs/config.json \
    --ckpt model/hawkeye.pth \
    --video-path data/videos/activitynet \
    --data-path data/test-anno/anetc-recursive_grounding.json \
    --save-path outputs/anetc-recursive_grounding-${max_turns}_turns.jsonl \
    > outputs/anetc-recursive_grounding-${max_turns}_turns.log 2>&1 &
wait

# test on nextgqa
python tasks/test_recursive_grounding.py --max-turns ${max_turns} --config configs/config.json \
    --ckpt model/hawkeye.pth \
    --video-path data/videos/nextqa \
    --data-path data/test-anno/nextgqa-recursive_grounding.json \
    --save-path outputs/nextgqa-recursive_grounding-${max_turns}_turns.jsonl \
    > outputs/nextgqa-recursive_grounding-${max_turns}_turns.log 2>&1 &
wait
