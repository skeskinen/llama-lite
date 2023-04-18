# llama-lite

Lightweight version of Llama transformer model intended for generating simple and fast sentence embeddings.

This is an experimental project, and the quality of the embeddings might not be good enough for your application. For better quality embeddings, check [Sentence Transformers](https://sbert.net/). This project might be for you if you want to do inference on CPU and don't like running Python.

llama-lite is a 134m parameter transformer model with hidden dim/embedding width of 768.
After 4bit quantization the model is 85MB and runs in 1.5ms per token on Ryzen 5 5600X.
This size and performance together with the c api of llama.cpp could make for a pretty nice local embeddings service.

Interesting parts of this repo:
1. model_creation has the python code for creating the model from scratch. You can train a "reasonable" model with pretty low amount of compute.
2. This repo forks ggerganov/llama.cpp and modifies it to work on the new small architecture
3. In examples there are new embeddings binaries, notably embeddings-server which starts a "toy" server that serves embeddings on port 8080.


Basic operation, just download the quantized testing weights
```
make
pip3 install -r requirements.txt
python3 model_creation/download_q4_weights.py
./embeddings-server

in another terminal:
python3 examples/embeddings-server-client.py

#example output of embeddings-server-client.py:
Starting with a test query "Should I get health insurance?"
Closest texts:
1. Should I sign up for Medicare Part B if I have Veterans' Benefits? (similarity score: 0.5214)
2. How do I get a replacement Medicare card? (similarity score: 0.2384)
3. How do I terminate my Medicare Part B (medical insurance)? (similarity score: 0.1932)
Enter a text to find similar texts (enter 'q' to quit): What is Medicare?
Closest texts:
1. What is Medicare and who can get it? (similarity score: 0.7388)
2. What is TRICARE ? (similarity score: 0.7163)
3. What is the monthly premium for Medicare Part B? (similarity score: 0.6744)
Enter a text to find similar texts (enter 'q' to quit): q
```

Weights are stored on huggingface: [skeskinen/llama-lite-134m](https://huggingface.co/skeskinen/llama-lite-134m/tree/main)

TODO:
- Train a good GPT model with a lot of data and then implement the contrastive pre-training from this [OpenAI text embedding paper](https://cdn.openai.com/papers/Text_and_Code_Embeddings_by_Contrastive_Pre_Training.pdf). 
- Make the server more robust, safe, etc.
- Probably a lot of other improvements. This was hacked together quite quickly.


Training the current, poor, pre-computed weights took ~15 mins on rtx 3090.

Making your own model:
```
make
pip3 install -r requirements.txt
cd model_creation
python3 model.py # this downloads the alpaca dataset & tokenizer and runs the training
sh ggml-conversion.sh
```


Benchmarks
==========================
examples/mteb-benchmark.py can be used to run mteb embeddings benchmark suite.
The results are in mteb-results folder. In the result jsons, the final score is the cos_sim.spearman value.
For reference scores check. https://huggingface.co/spaces/mteb/leaderboard
This model would be clear last on the leaderboard. Even Llama 7B does very poorly, so something like the contrastive training from [the OpenAI paper](https://cdn.openai.com/papers/Text_and_Code_Embeddings_by_Contrastive_Pre_Training.pdf) is probably a necessity for real applications.