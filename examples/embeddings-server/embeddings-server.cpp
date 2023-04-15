#include <iostream>
#include <string>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include "common.h"
#include "llama.h"

constexpr int PORT = 8080;

std::string receive_string(int socket) {
    char buffer[1 << 15] = {0};
    ssize_t bytes_received = read(socket, buffer, sizeof(buffer));
    return std::string(buffer, bytes_received);
}

void send_floats(int socket, const float* floats, int size) {
    send(socket, floats, size * sizeof(float), 0);
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.model = "models/lite/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    params.embedding = true;
    params.n_ctx = 2048;

    llama_context * ctx;

    // load the model
    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx      = params.n_ctx;
        lparams.n_parts    = params.n_parts;
        lparams.seed       = params.seed;
        lparams.f16_kv     = params.memory_f16;
        lparams.logits_all = params.perplexity;
        lparams.use_mmap   = params.use_mmap;
        lparams.use_mlock  = params.use_mlock;
        lparams.embedding  = params.embedding;

        ctx = llama_init_from_file(params.model.c_str(), lparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }

    int n_past = 0;

    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        std::cerr << "Socket creation failed" << std::endl;
        return -1;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        std::cerr << "Bind failed" << std::endl;
        return -1;
    }

    if (listen(server_fd, 1) < 0) {
        std::cerr << "Listen failed" << std::endl;
        return -1;
    }

    std::cout << "Server running on port " << PORT << std::endl;

    while(true) {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0) {
            std::cerr << "Accept failed" << std::endl;
            return -1;
        }

        std::string received_string = receive_string(new_socket);
        std::cout << "Received: " << received_string << std::endl;

        auto embd_inp = ::llama_tokenize(ctx, received_string, false);
        if (embd_inp.size() > 0) {
            if (llama_eval(ctx, embd_inp.data(), embd_inp.size(), n_past, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }
        }

        const auto embeddings = llama_get_embeddings(ctx);

        send_floats(new_socket, embeddings, llama_n_embd(ctx));

        close(new_socket);
    }
    close(server_fd);

    return 0;
}
