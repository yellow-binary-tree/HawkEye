folder=$1
ckpt=$2

# test on mvbench
python tasks/test.py --tasks "MVBench" --config ${folder}/config.json \
    --ckpt ${folder}/ckpt_${ckpt}.pth --data-dir data \
    --save-path ${folder}/MVBench-${ckpt}.jsonl \
    > ${folder}/MVBench-${ckpt}.log 2>&1
wait

# test on NExT-QA
python tasks/test.py --tasks "NExTQA" --config ${folder}/config.json \
    --ckpt ${folder}/ckpt_${ckpt}.pth --data-dir data \
    --save-path ${folder}/NExTQA-${ckpt}.jsonl \
    > ${folder}/NExTQA-${ckpt}.log 2>&1
wait


# test on TVQA
python tasks/test.py --tasks "TVQA" --config ${folder}/config.json \
    --ckpt ${folder}/ckpt_${ckpt}.pth --data-dir data \
    --save-path ${folder}/TVQA-${ckpt}.jsonl \
    > ${folder}/TVQA-${ckpt}.log 2>&1
wait


# test on STAR
python tasks/test.py --tasks "STAR" --config ${folder}/config.json \
    --ckpt ${folder}/ckpt_${ckpt}.pth --data-dir data \
    --save-path ${folder}/STAR-${ckpt}.jsonl \
    > ${folder}/STAR-${ckpt}.log 2>&1
wait
