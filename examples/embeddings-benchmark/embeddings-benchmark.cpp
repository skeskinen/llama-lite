#include "common.h"
#include "llama.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;


float dot_product(vector<float> &v1, vector<float> &v2) {
    float result = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

float magnitude(vector<float> &v) {
    float result = 0.0;
    for (size_t i = 0; i < v.size(); i++) {
        result += pow(v[i], 2);
    }
    return sqrt(result);
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.model = "models/lite/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    params.embedding = true;

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

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

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    int n_past = 0;

    // Add a space in front of the first character to match OG llama tokenizer behavior
    //params.prompt.insert(0, 1, ' ');


    vector<std::string> prompts = {
"That is a happy person",
"That is a happy dog",
"That is a very happy person",
"Today is a sunny day",
    };
    const int N = prompts.size();
    const int n_embd = llama_n_embd(ctx);
    vector<vector<float>> res(N);
    // determine newline token
    //auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    for( int k = 0 ; k < N; k++) {
        // tokenize the prompt
        auto embd_inp = ::llama_tokenize(ctx, prompts[k], false);
        if (embd_inp.size() > 0) {
            if (llama_eval(ctx, embd_inp.data(), embd_inp.size(), n_past, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }
        }

        const auto embeddings = llama_get_embeddings(ctx);
  //      printf("[");
        for (int i = 0; i < n_embd; i++) {
            res[k].push_back(embeddings[i]);
//            printf("%f, ", embeddings[i]);
        }
    //    printf("]\n");
    }

    vector<vector<float>> similarity_matrix(N, vector<float>(N));
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            float dot_prod = dot_product(res[i], res[j]);
            float mag1 = magnitude(res[i]);
            float mag2 = magnitude(res[j]);
            cout << prompts[i] << " - " << prompts[j] << ": " << dot_prod / (mag1 * mag2) << endl;
            similarity_matrix[i][j] = dot_prod / (mag1 * mag2);
        }
    }

    // Print the similarity matrix
    /*
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << similarity_matrix[i][j] << " ";
        }
        cout << endl;
    }
    */

    llama_print_timings(ctx);
    llama_free(ctx);

    return 0;
}
