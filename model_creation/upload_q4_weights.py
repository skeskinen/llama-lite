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