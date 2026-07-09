from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

# dataset = load_dataset("datasetfileexists")

model_name = "basemodel/"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

training_args = SFTConfig(
    otput_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

trainer.train() 
