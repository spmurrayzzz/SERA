echo "MODEL_NAME=GLM-4,5-Air"
echo "TP_SIZE=$1"
echo "PORT=$2"
echo "RANDOM_SEED=$3"
python3 -m sglang.launch_server \
        --model-path zai-org/GLM-4.5-Air \
        --tp-size $1 \
        --tool-call-parser glm45  \
        --mem-fraction-static 0.87 \
        --disable-shared-experts-fusion \
        --host 0.0.0.0 \
        --port $2 \
        --random-seed $3 \
        --context-length 128000 \
        --allow-auto-truncate