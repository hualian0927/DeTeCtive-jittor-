Model_PATH="./runs/model_best.pth"
DATABASE_PATH="database/deepfake"

python infer.py \
    --database_path ${DATABASE_PATH} \
    --model_path ${Model_PATH} \
    --K 5 \
    --text "I really want someone to change my view on this, since everyone I know are frowning on me for thinking this way."