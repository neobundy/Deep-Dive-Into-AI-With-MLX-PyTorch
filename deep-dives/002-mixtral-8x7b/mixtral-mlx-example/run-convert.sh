MODEL_PATH="/Users/wankyuchoi/cwk-llm-models/Mixtral-8x7B-Instruct-v0.1"
MLX_PATH="/Users/wankyuchoi/cwk-llm-models/Mixtral-8x7B-Instruct-v0.1-8bit-mlx"

python convert.py --torch-path $MODEL_PATH --quantize --q-bits 8 --mlx-path $MLX_PATH