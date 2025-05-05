/*
 * The MIT License (MIT)
 *
 * Copyright 2024-present Victor Hogeweij
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * Author:          Victor Hogeweij <Hoog-V>
 *
 */

#ifndef SIMPLELLAMA_HPP
#define SIMPLELLAMA_HPP
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <llama.h>

/*
 * Settings for the model inference,
 * Such as usage of gpu, threads etc.
 */
struct simplellama_model_params
{
    /* How many threads to use ?, default is 4*/
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());

    /* Number of gpu-layers to export to gpu, default = all 
    *  More layers means more vram usage! But better performance
    */
    int32_t n_gpu_layers = 999;

    /* Whether or not to give a verbose prompt */
    bool verbose_prompt = false;
    /* Whether or not to use gpu, would highly suggest to leave enabled */
    bool use_gpu = true;
    /* whether to use flash attention [EXPERIMENTAL] */
    bool flash_attn = false;

    /* This is your name, well the LLM thinks who you are */
    std::string person = "Georgi";
    /* This is the name of the LLM which will be given in default prompt */
    std::string bot_name = "LLaMA";
    /* The model file .gguf location+filename, for the example cmake will download phi-2 */
    std::string model_llama = "";
    /* Custom prompt for model init, if no prompt given, it will use default */
    std::string prompt = "";
    std::string path_session = ""; // path to file for saving/loading model eval state
};

class SimpleLLama
{
public:
    /**
     * @brief Construct a new llama wrapper object
     *
     * @param params A struct containing configuration for model inference; Runtime as well as behavioural settings
     */
    SimpleLLama(simplellama_model_params params);

    /**
     * @brief Initialize llama by initializing a talk session with a initializer response
     */
    void init();

    /**
     * @brief Run inference (this reuses the initialised session)
     *
     * @param input_text user_prompt, to react on
     * @return std::string The response from the llama model
     */
    std::string do_inference(std::string &input_text);

    /**
     * @brief Destroy the llama wrapper object
     *
     */
    ~SimpleLLama()
    {
        llama_perf_sampler_print(smpl);
        llama_perf_context_print(m_ctx);
        llama_sampler_free(smpl);
        llama_batch_free(m_batch);
        llama_free(m_ctx);
    }

private:
    /* This is used to store model details (tokens, architecture, etc) about the to be loaded model */
    llama_model *model_llama;
    llama_vocab * m_vocab;

    const llama_vocab * m_vocab_llama;
    /* This is the context that is used with the model and session
     * When used with the session tokens, it can make the model react on previous responses (learn from it, kinda)
     */
    llama_context *m_ctx;

    llama_sampler *smpl;
    /* Configuration parameters for the model and context */
    /* The default settings should be fine, and give a good starting ground */
    llama_context_params m_lcparams = llama_context_default_params();
    llama_model_params m_lmparams = llama_model_default_params();

    // /* A copy of the parameters passed to the constructor */
    // /* The settings are used multiple times during inference and configuration */
    simplellama_model_params m_params;

    /* Session tokens which are used to help do inference on text prompts */
    std::vector<llama_token> m_session_tokens;

    // /* Antiprompts are used to prevent the model from going mayhem when prompt is empty or invalid text is put in to the prompt */
    std::vector<std::string> m_antiprompts;

    // /* Tokens of prompt that gets infered */
    std::vector<llama_token> m_embd;
    struct gpt_vocab
    {
        using id = int32_t;
        using token = std::string;

        std::map<token, id> token_to_id;
        std::map<id, token> id_to_token;
        std::vector<std::string> special_tokens;

        void add_special_token(const std::string &token);
    };

    std::vector<gpt_vocab::id> m_embd_inp;

    /* Batch, this is used for decoding the response of the model*/
    llama_batch m_batch;

    /* Sessions can be loaded/unloaded to speed up initializing of the model for inference */
    /* This variable contains the path to the file where last session was saved to */
    std::string m_path_session;
    bool m_need_to_save_session;

    int m_n_past;
    int m_n_prev;

    int m_n_session_consumed;

    const int m_voice_id = 2;

    int m_n_keep;

    int m_n_ctx;

    bool m_is_running;

      /* Prepare the batch, so that it can be used later for decoding responses */
    void prepare_batch(std::vector<llama_token> &tokens, int n_past)
    {
        {
            m_batch.n_tokens = tokens.size();

            for (int i = 0; i < m_batch.n_tokens; i++)
            {
                m_batch.token[i] = tokens[i];
                m_batch.pos[i] = n_past + i;
                m_batch.n_seq_id[i] = 1;
                m_batch.seq_id[i][0] = 0;
                m_batch.logits[i] = i == m_batch.n_tokens - 1;
            }
        }
    }
};

#endif