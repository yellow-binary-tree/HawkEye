# test on mvbench
python tasks/test.py --tasks "MVBench" --config configs/config.json \
    --ckpt model/hawkeye.pth --data-dir data \
    --save-path outputs/MVBench.jsonl \
    > outputs/MVBench.log 2>&1 &
wait

# test on NExT-QA
python tasks/test.py --tasks "NExTQA" --config configs/config.json \
    --ckpt model/hawkeye.pth --data-dir data \
    --save-path outputs/NExTQA.jsonl \
    > outputs/NExTQA.log 2>&1 &
wait


# test on TVQA
python tasks/test.py --tasks "TVQA" --config configs/config.json \
    --ckpt model/hawkeye.pth --data-dir data \
    --save-path outputs/TVQA.jsonl \
    > outputs/TVQA.log 2>&1 &
wait


# test on STAR
python tasks/test.py --tasks "STAR" --config configs/config.json \
    --ckpt model/hawkeye.pth --data-dir data \
    --save-path outputs/STAR.jsonl \
    > outputs/STAR.log 2>&1 &
wait