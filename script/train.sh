# 建议放在项目根目录运行
DATA_PATH="/opt/LLM_detect_data"
SAVE_DIR="./runs"  # 建议使用相对路径或你指定的绝对路径

# Jittor 单卡训练配置
# device_num: 设为 1
# per_gpu_batch_size: A100 显存很大(40G/80G)，可以尝试设大一点，比如 32 或 64
python train_classifier.py \
    --device_num 1 \
    --per_gpu_batch_size 32 \
    --total_epoch 50 \
    --lr 2e-5 \
    --warmup_steps 2000 \
    --model_name princeton-nlp/unsup-simcse-roberta-base \
    --dataset deepfake \
    --path ${DATA_PATH}/Deepfake/cross_domains_cross_models \
    --name deepfake-roberta-base \
    --freeze_embedding_layer \
    --database_name train \
    --test_dataset_name test \
    --savedir ${SAVE_DIR}