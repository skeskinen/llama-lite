
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from huggingface_hub import HfApi
import os

os.chdir(os.path.dirname(__file__))
api = HfApi()

api.upload_file(

    path_or_fileobj="../models/lite/ggml-model.bin",

    path_in_repo="ggml-model.bin",

    repo_id="skeskinen/llama-lite-134m",

    repo_type="model",

)

tokenizer = LlamaTokenizer.from_pretrained("./output",
                                           add_bos_token=True,
                                           add_eos_token=True)
model = LlamaForCausalLM.from_pretrained("./output")

# Save the model and tokenizer to the Hugging Face Hub
model.push_to_hub("skeskinen/llama-lite-134m")
tokenizer.push_to_hub("skeskinen/llama-lite-134m")
