# Copyright © 2023 Apple Inc.

import json
from pathlib import Path
from typing import List

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from models import LoRALinear, Model, ModelArgs
from sentencepiece import SentencePieceProcessor

import streamlit as st


class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "▁"
        assert self._model.vocab_size() == self._model.get_piece_size()

    def encode(self, s: str, eos: bool = False) -> List[int]:
        toks = [self._model.bos_id(), *self._model.encode(s)]
        if eos:
            toks.append(self.eos_id)
        return toks

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    def decode(self, t: List[int]) -> str:
        out = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + out
        return out

    @property
    def vocab_size(self) -> int:
        return self._model.vocab_size()


def generate(model, prompt, tokenizer, num_tokens, temp):
    print(prompt, end="", flush=True)
    prompt = mx.array(tokenizer.encode(prompt))
    def generate_step():
        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        logits, cache = model(prompt[None])
        y = sample(logits[:, -1, :])
        yield y

        while True:
            logits, cache = model(y[:, None], cache)
            y = sample(logits.squeeze(1))
            yield y

    tokens = []
    for token, _ in zip(generate_step(), range(num_tokens)):
        tokens.append(token)

        if (len(tokens) % 10) == 0:
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s, flush=True)
            st.write(s)
            tokens = []

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)

def load_model(folder: str, dtype=mx.float16):
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "params.json", "r") as f:
        config = json.loads(f.read())
        if config.get("vocab_size", -1) < 0:
            config["vocab_size"] = tokenizer.vocab_size
        model_args = ModelArgs(**config)
    weights = mx.load(str(model_path / "weights.npz"))
    weights = tree_unflatten(list(weights.items()))
    weights = tree_map(lambda p: p.astype(dtype), weights)
    model = Model(model_args)
    model.update(weights)
    return model, tokenizer

# Streamlit widgets for the arguments
model_path = st.text_input('Model path', '/Users/wankyuchoi/cwk-llm-models/mistral-7B-v0.1-mlx')
adapter_file = st.text_input('Adapter path', './adapters.npz')
num_tokens = st.number_input('Number of tokens', min_value=1, max_value=500, value=50)
temp = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.8)
prompt = st.text_input('Prompt', 'Q: What is ReLu in mlx?')


# Load the model and tokenizer
model, tokenizer = load_model(model_path)
print("Loading lora...")
model.load_weights(adapter_file)
# Call the generate function with the values from the widgets
if st.button('Generate'):
    st.info(f"Prompt: {prompt}")
    generate(model, prompt, tokenizer, num_tokens, temp)
