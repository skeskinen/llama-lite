import socket
import struct
import numpy as np
from mteb import MTEB

RUN_7B_MODEL = False

def embeddings_from_local_server(s):
    host = "localhost"
    port = 8080

    n_embeds = 4096 if RUN_7B_MODEL else 768

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        sock.sendall(s.encode())
        data = sock.recv(n_embeds*4)
    floats = struct.unpack('f' * n_embeds, data)
    return floats


class MyModel():
    def encode(self, sentences, batch_size=32, **kwargs):
        results = []
        for s in sentences:
            results.append(np.array(embeddings_from_local_server(s)))
        return results

model = MyModel()

#Run the Semantic Textual Similarity tests
evaluation = MTEB(tasks=[
    "BIOSSES",
    #"STS12",
    #"STS13",
    #"STS14",
    #"STS15",
    #"STS16",
    "STS17",
    #"STSBenchmark"
    ])
output_folder = "mteb-results"
if RUN_7B_MODEL:
    output_folder += "-7B"
evaluation.run(model, output_folder=output_folder, eval_splits=["test"], task_langs=["en"])