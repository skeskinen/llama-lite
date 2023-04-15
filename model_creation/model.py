from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Create a custom LlamaConfig
config = LlamaConfig(
    vocab_size=32000,
    hidden_size=768,
    intermediate_size=2048, # To match the original llama code, intermediate size is calculated like this: 256 * (((768*8/3) + 256 - 1) //256) = 2048
    num_hidden_layers=12,  # Adjust as needed to balance model size and performance
    num_attention_heads=6, # hidden_size // 128
    hidden_act="silu",
    max_position_embeddings=2048,
    initializer_range=0.02,
    rms_norm_eps=1e-12,
    use_cache=True,
    tie_word_embeddings=False
)

# Total params should be roughly:
# 12*(768*768*4+2048*768*3+768*2) + (32000 * 768 * 2) = 134M

# Create a new LLaMA model with the custom configuration
model = LlamaForCausalLM(config=config)

tokenizer = LlamaTokenizer.from_pretrained("aleksickx/llama-7b-hf",
                                           add_bos_token=True,
                                           add_eos_token=True)
tokenizer.pad_token_id = (1) # eos, other option is 0 which stands for unk

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048)

dataset = load_dataset("tatsu-lab/alpaca")
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./tmp-output",
    overwrite_output_dir=True,
    num_train_epochs=1, 
    per_device_train_batch_size=4,  # Adjust as needed
    warmup_steps=100,
    save_steps=3000,
    save_total_limit=2,
    optim="adamw_torch",
)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer,
            mlm=False,  # Set to False for causal language modeling       
            pad_to_multiple_of=8,
            return_tensors='pt',
        ), 
)

# Train the model
trainer.train()

# Save the model and tokenizer to disk
model.save_pretrained("./output")
tokenizer.save_pretrained("./output")

# Save the model and tokenizer to the Hugging Face Hub
model.push_to_hub("skeskinen/llama-lite-134m")
tokenizer.push_to_hub("skeskinen/llama-lite-134m")
