MODEL_PATH="/Users/wankyuchoi/cwk-llm-models/Mistral-7B-Instruct-v0.2-mlx"
MAX_TOKENS=200
TEMP=0.7
PROMPT="<<SYS>>Your name is Menny, a cynical teenager AI assistant.<</SYS>>[INST] Who are you? [/INST]"

python mistral.py --model-path $MODEL_PATH \
    --max-tokens $MAX_TOKENS --temp $TEMP \
    --prompt "$PROMPT"