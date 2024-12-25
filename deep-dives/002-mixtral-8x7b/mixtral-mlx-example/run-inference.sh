MODEL_PATH="/Users/wankyuchoi/cwk-llm-models/Mixtral-8x7B-Instruct-v0.1-mlx"
MAX_TOKENS=200
TEMP=0.7
PROMPT="<<SYS>>Your name is Menny, a cynical teenager AI assistant.<</SYS>>[INST] Who are you? [/INST]"

python mixtral.py --model-path $MODEL_PATH \
    --max-tokens $MAX_TOKENS --temp $TEMP \
    --prompt "$PROMPT"