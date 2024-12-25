MODEL_PATH="/Users/wankyuchoi/cwk-llm-models/llama-2-7b-chat-mlx"
MAX_TOKENS=200
WRITE_EVERY=100
TEMP=0.7
PROMPT="<<SYS>>Your name is Menny, a cynical teenager AI assistant.<</SYS>>[INST] Who are you? [/INST]"

python llama.py --model-path $MODEL_PATH \
    --max-tokens $MAX_TOKENS --write-every $WRITE_EVERY --temp $TEMP \
    --prompt "$PROMPT"