from huggingface_hub import hf_hub_download
import os

os.chdir(os.path.dirname(__file__))

hf_hub_download(repo_id="skeskinen/llama-lite-134m", filename="ggml-model.bin", repo_type="model", local_dir="../models/lite", local_dir_use_symlinks=False)