DATA_PATH="/opt/LLM_detect_data"
# 注意：这里需要指向你训练好的模型路径 (train.sh 跑完后生成的)
Model_PATH="./runs/model_best.pth" 

# deepfake 数据库生成
python gen_database.py \
    --device_num 1 \
    --batch_size 128 \
    --model_name princeton-nlp/unsup-simcse-roberta-base \
    --mode deepfake \
    --database_path ${DATA_PATH}/Deepfake/cross_domains_cross_models \
    --database_name 'train' \
    --model_path ${Model_PATH} \
    --save_path database/deepfake