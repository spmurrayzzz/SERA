# --node-rank and --dist-init-addr should be filled in from env variables at runtime.

echo "MODEL_NAME=GLM-4.6"
echo "PORT=$1"
echo "RANDOM_SEED=$2"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 &&
python3 -m sglang.launch_server \
    --model-path zai-org/GLM-4.6 \
    --host 0.0.0.0 \
    --tool-call-parser glm45 \
    --mem-fraction-static 0.92 \
    --disable-shared-experts-fusion \
    --served-model-name GLM-4.6 \
    --port $1 \
    --tp-size 16 \
    --dp-size 1 \
    --nnodes 2 \
    --node-rank  \
    --dist-init-addr  \
    --trust-remote-code \
    --random-seed $2 \
    --context-length 65536 \
    --allow-auto-truncate