export MASTER_PORT=$((12000 + $RANDOM % 20000))
NNODE=1
NUM_GPUS=8

OUTPUT_DIR=$1
mkdir -vp $OUTPUT_DIR

torchrun  --nnodes=${NNODE} --nproc_per_node=${NUM_GPUS}  --master_port=${MASTER_PORT} \
    tasks/train_it.py scripts/train/config_7b_stage3.py \
    output_dir ${OUTPUT_DIR} \
    freeze_dataset_folder ${OUTPUT_DIR}/training_data \
    > ${OUTPUT_DIR}/train.log 2>&1 &

wait
