import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, IntervalStrategy,EvalPrediction, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from pathlib import Path
import json


MODEL = "microsoft/phi-2"
MODEL_NAME = "Phi-2"
DATA_FOLDER = './data'
OUTPUT_FOLDER = './lora'

NUM_EPOCHS = 400
LEARNING_RATE = 0.001

LORA_CONFIG = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['q_proj', 'k_proj', 'v_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift the input to the right for the labels
        labels = inputs["input_ids"][:, 1:].contiguous()
        # Ignore the last token for the logits
        logits = logits[:, :-1, :].contiguous()

        # Mask padding tokens
        attention_mask = inputs["attention_mask"][:, 1:].contiguous()
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, model.config.vocab_size)[active_loss]
        active_labels = labels.view(-1)[active_loss]

        # Calculate the loss
        loss = cross_entropy(active_logits, active_labels)
        ntoks = active_loss.sum()

        return (loss, outputs) if return_outputs else loss


def my_compute_metrics(eval_output):
    logits, labels = eval_output

    # Convert numpy arrays to PyTorch tensors
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    # Clip the values of the logits to avoid numerical instability
    logits = torch.clamp(logits, min=-10, max=10)

    # Shift labels to ignore padding token id for cross-entropy loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Add a small constant to avoid division by zero
    avg_loss = loss / (shift_labels.ne(tokenizer.pad_token_id).sum() + 1e-9)
    perplexity = torch.exp(avg_loss)

    # Calculate and return eval_loss
    eval_loss = avg_loss.item()
    return {"eval_perplexity": perplexity.item(), "eval_loss": eval_loss}


def load_model(model, device):
    base_model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return base_model, tokenizer


def inspect_model_architecture(model, model_name):

    print(f"\n========= Model Architecture: {model_name} =========\n\n")

    print(model)
    for name, param in model.named_parameters():
        print(name, param.shape)


def inspect_trainable_parameters(model, model_name):
    print(f"\n========= Trainable Parameters: {model_name} =========\n\n")
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}"
    )


def generate(model, tokenizer, prompt, max_tokens=100):
    batch = tokenizer(f"#context\n\n{prompt}\n\n#response\n\n", return_tensors='pt')

    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, do_sample=True, temperature=0.7, max_new_tokens=max_tokens)

    print(tokenizer.decode(output_tokens[0], skip_special_tokens=True), flush=True)


def load_dataset(data_folder, tokenizer):
    names = ("train", "valid", "test")
    datasets = {}
    for name in names:
        with open(Path(data_folder) / f"{name}.jsonl", 'r') as file:
            lines = file.readlines()
            data = [json.loads(line) for line in lines]
            # Each data entry is a dictionary with 'text' key
            texts = [item['text'] for item in data]
            # Convert the texts into tensors
            encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
            datasets[name] = TextDataset(encodings)
    return datasets


def train(model, tokenizer, datasets):

    training_args = TrainingArguments(
        output_dir=OUTPUT_FOLDER,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="eval_perplexity",
        # logging_strategy=IntervalStrategy.STEPS,  # Log after every step
        # logging_steps=1,  # Adjust this to set the logging frequency
        # logging_first_step=True,  # Log the first step to make sure logging works
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        tokenizer=tokenizer,
        compute_metrics=my_compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()


def load_model_with_adapter(peft_model_id, device):
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True, return_dict=True, load_in_8bit=False, device_map='auto').to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    lora_model = PeftModel.from_pretrained(model, peft_model_id)
    return lora_model, tokenizer


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    base_model, tokenizer = load_model(MODEL, device)

    config = LORA_CONFIG

    # Uncomment the following lines to train Tenny yourself
    # inspect_model_architecture(base_model, MODEL_NAME)
    # inspect_trainable_parameters(base_model, MODEL_NAME)
    # lora_model = get_peft_model(base_model, config)
    # inspect_model_architecture(lora_model, f"{MODEL_NAME} + LORA")
    # inspect_trainable_parameters(lora_model, f"{MODEL_NAME} + LORA")
    # lora_model.print_trainable_parameters() # The above function implemented here for illustration purposes. You can use PEFT's print_trainable_parameters() instead.

    # datasets = load_dataset(DATA_FOLDER, tokenizer)
    #
    # train(lora_model, tokenizer, datasets)
    # lora_model.save_pretrained(OUTPUT_FOLDER)
    # lora_model.push_to_hub("your-name/folder") # You can also push the model to the Hugging Face Hub if you want.

    # When you have to save the configuration and tokenizer of the base model
    # config = base_model.config
    # config.save_pretrained(OUTPUT_FOLDER)
    # tokenizer.save_pretrained(OUTPUT_FOLDER)

    inference_model, tokenizer = load_model_with_adapter(OUTPUT_FOLDER, device)
    prompt = "I love my new MacBook Pro. What do you think?"

    generate(inference_model, tokenizer, prompt, max_tokens=100)
