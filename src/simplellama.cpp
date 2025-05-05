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

#include "simplellama.hpp"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <regex>
#include <sstream>
#include <SDL2/SDL.h>

bool sdl_poll_events()
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        switch (event.type)
        {
        case SDL_QUIT:
        {
            return false;
        }
        break;
        default:
            break;
        }
    }

    return true;
}

const std::string k_prompt_llama = R"(Text transcript of a never ending dialog, where {0} interacts with an AI assistant named {1}.
{1} is helpful, kind, honest, friendly, good at writing and never fails to answer {0}’s requests immediately and with details and precision.
There are no annotations like (30 seconds passed...) or (to himself), just what {0} and {1} say aloud to each other.
The transcript only includes text, it does not include markup like HTML and Markdown.
{1} responds with short and concise answers.

{0}{4} Hello, {1}!
{1}{4} Hello {0}! How may I help you today?
{0}{4} What time is it?
{1}{4} It is {2} o'clock.
{0}{4} What year is it?
{1}{4} We are in {3}.
{0}{4} What is a cat?
{1}{4} A cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae.
{0}{4} Name a color.
{1}{4} Blue
{0}{4})";

std::string replace(const std::string &s, const std::string &from, const std::string &to)
{
    std::string result = s;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos)
    {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    return result;
}

static std::vector<llama_token> llama_tokenize(struct llama_context *ctx, const std::string &text, bool add_bos)
{
    
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // upper limit for the number of tokens
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_bos, false);
    if (n_tokens < 0)
    {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_bos, false);
        GGML_ASSERT(check == -n_tokens);
    }
    else
    {
        result.resize(n_tokens);
    }
    return result;
}

static std::string llama_token_to_piece(const struct llama_context *ctx, llama_token token)
{
    std::vector<char> result(8, 0);
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_tokens = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, false);
    if (n_tokens < 0)
    {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, false);
        GGML_ASSERT(check == -n_tokens);
    }
    else
    {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

std::string construct_initial_prompt_for_llama(simplellama_model_params &params)
{
    const std::string chat_symb = ":";
    // construct the initial prompt for LLaMA inference
    std::string prompt_llama = params.prompt.empty() ? k_prompt_llama : params.prompt;

    // need to have leading ' '
    prompt_llama.insert(0, 1, ' ');

    prompt_llama = ::replace(prompt_llama, "{0}", params.person);
    prompt_llama = ::replace(prompt_llama, "{1}", params.bot_name);

    {
        // get time string
        std::string time_str;
        {
            time_t t = time(0);
            struct tm *now = localtime(&t);
            char buf[128];
            strftime(buf, sizeof(buf), "%H:%M", now);
            time_str = buf;
        }
        prompt_llama = ::replace(prompt_llama, "{2}", time_str);
    }

    {
        // get year string
        std::string year_str;
        {
            time_t t = time(0);
            struct tm *now = localtime(&t);
            char buf[128];
            strftime(buf, sizeof(buf), "%Y", now);
            year_str = buf;
        }
        prompt_llama = ::replace(prompt_llama, "{3}", year_str);
    }

    prompt_llama = ::replace(prompt_llama, "{4}", chat_symb);
    return prompt_llama;
}

SimpleLLama::SimpleLLama(simplellama_model_params params)
{
    m_params = params;
    llama_backend_init();

    if (!params.use_gpu)
    {
        m_lmparams.n_gpu_layers = 0;
    }
    else
    {
        m_lmparams.n_gpu_layers = params.n_gpu_layers;
    }

    model_llama = llama_model_load_from_file(params.model_llama.c_str(), m_lmparams);
    if (!model_llama)
    {
        fprintf(stderr, "No llama.cpp model specified. Please provide using -ml <modelfile>\n");
        exit(1);
    }

    // tune these to your liking
    m_lcparams.n_ctx = 2048;
    m_lcparams.n_threads = params.n_threads;
    m_lcparams.flash_attn = params.flash_attn;
    m_lcparams.no_perf = false;
    m_ctx = llama_init_from_model(model_llama, m_lcparams);
    m_vocab_llama = llama_model_get_vocab(model_llama);
}

void SimpleLLama::init()
{
    std::string prompt_llama = construct_initial_prompt_for_llama(m_params);
    // init sampler
    const float top_k = 5;
    const float top_p = 0.80f;
    const float temp = 0.30f;

    const int seed = 0;

    auto sparams = llama_sampler_chain_default_params();

    smpl = llama_sampler_chain_init(sparams);

    if (temp > 0.0f)
    {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    }
    else
    {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    m_batch = llama_batch_init(llama_n_ctx(m_ctx), 0, 1);

    m_path_session = m_params.path_session;

    m_embd_inp = ::llama_tokenize(m_ctx, prompt_llama, true);

    if (!m_path_session.empty())
    {
        fprintf(stderr, "%s: attempting to load saved session from %s\n", __func__, m_path_session.c_str());

        // fopen to check for existing session
        FILE *fp = std::fopen(m_path_session.c_str(), "rb");
        if (fp != NULL)
        {
            std::fclose(fp);

            m_session_tokens.resize(llama_n_ctx(m_ctx));
            size_t n_token_count_out = 0;

            if (!llama_state_load_file(m_ctx, m_path_session.c_str(), m_session_tokens.data(), m_session_tokens.capacity(), &n_token_count_out))
            {
                fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, m_path_session.c_str());
                exit(1);
            }
            m_session_tokens.resize(n_token_count_out);
            for (size_t i = 0; i < m_session_tokens.size(); i++)
            {
                m_embd_inp[i] = m_session_tokens[i];
            }

            fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int)m_session_tokens.size());
        }
        else
        {
            fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
        }
    }

    prepare_batch(m_embd_inp, 0);

    if (llama_decode(m_ctx, m_batch))
    {
        fprintf(stderr, "%s : failed to decode\n", __func__);
        exit(1);
    }

    if (m_params.verbose_prompt)
    {
        fprintf(stdout, "\n");
        fprintf(stdout, "%s", prompt_llama.c_str());
        fflush(stdout);
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (m_session_tokens.size())
    {
        for (llama_token id : m_session_tokens)
        {
            if (n_matching_session_tokens >= m_embd_inp.size() || id != m_embd_inp[n_matching_session_tokens])
            {
                break;
            }
            n_matching_session_tokens++;
        }
        if (n_matching_session_tokens >= m_embd_inp.size())
        {
            fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
        }
        else if (n_matching_session_tokens < (m_embd_inp.size() / 2))
        {
            fprintf(stderr, "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, m_embd_inp.size());
        }
        else
        {
            fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, m_embd_inp.size());
        }
    }

    // HACK - because session saving incurs a non-negligible delay, for now skip re-saving session
    // if we loaded a session with at least 75% similarity. It's currently just used to speed up the
    // initial prompt so it doesn't need to be an exact match.
    m_need_to_save_session = !m_path_session.empty() && n_matching_session_tokens < (m_embd_inp.size() * 3 / 4);

    printf("%s : done! llama init succesfull \n", __func__);

    printf("\n");
    // printf("%s%s", m_params.person.c_str(), m_chat_symb.c_str());
    fflush(stdout);

    // text inference variables
    m_n_keep = m_embd_inp.size();
    m_n_ctx = llama_n_ctx(m_ctx);
    m_n_past = m_n_keep;
    m_n_prev = 64; // TODO arg
    m_n_session_consumed = !m_path_session.empty() && m_session_tokens.size() > 0 ? m_session_tokens.size() : 0;

    // reverse prompts for detecting when it's time to stop speaking
    m_antiprompts = {
        m_params.person + ":",
    };
}

std::string SimpleLLama::do_inference(std::string &text_heard)
{
    const std::vector<llama_token> tokens = llama_tokenize(m_ctx, text_heard.c_str(), false);

    if (tokens.empty())
    {
        // fprintf(stdout, "%s: Heard nothing, skipping ...\n", __func__);

        return "";
    }

    text_heard.insert(0, 1, ' ');
    text_heard += "\n" + m_params.bot_name + ":";

    m_embd = ::llama_tokenize(m_ctx, text_heard, false);
    // Append the new input tokens to the session_tokens vector
    if (!m_path_session.empty())
    {
        m_session_tokens.insert(m_session_tokens.end(), tokens.begin(), tokens.end());
    }

    // text inference
    bool done = false;
    std::string text_to_speak;
    while (true)
    {
        // predict
        if (m_embd.size() > 0)
        {
            if (m_n_past + (int)m_embd.size() > m_n_ctx)
            {
                m_n_past = m_n_keep;

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                m_embd.insert(m_embd.begin(), m_embd_inp.begin() + m_embd_inp.size() - m_n_prev, m_embd_inp.end());
                // stop saving session if we run out of context
                m_path_session = "";
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            // REVIEW
            if (m_n_session_consumed < (int)m_session_tokens.size())
            {
                size_t i = 0;
                for (; i < m_embd.size(); i++)
                {
                    if (m_embd[i] != m_session_tokens[m_n_session_consumed])
                    {
                        m_session_tokens.resize(m_n_session_consumed);
                        break;
                    }

                    m_n_past++;
                    m_n_session_consumed++;

                    if (m_n_session_consumed >= (int)m_session_tokens.size())
                    {
                        i++;
                        break;
                    }
                }
                if (i > 0)
                {
                    m_embd.erase(m_embd.begin(), m_embd.begin() + i);
                }
            }

            if (m_embd.size() > 0 && !m_path_session.empty())
            {
                m_session_tokens.insert(m_session_tokens.end(), m_embd.begin(), m_embd.end());
                m_n_session_consumed = m_session_tokens.size();
            }

            prepare_batch(m_embd, m_n_past);

            if (llama_decode(m_ctx, m_batch))
            {
                fprintf(stderr, "%s : failed to decode\n", __func__);
                exit(1);
            }
        }

        m_embd_inp.insert(m_embd_inp.end(), m_embd.begin(), m_embd.end());
        m_n_past += m_embd.size();

        m_embd.clear();

        if (done)
            break;

        {
            // out of user input, sample next token

            if (!m_path_session.empty() && m_need_to_save_session)
            {
                m_need_to_save_session = false;
                llama_state_save_file(m_ctx, m_path_session.c_str(), m_session_tokens.data(), m_session_tokens.size());
            }

            const llama_token id = llama_sampler_sample(smpl, m_ctx, -1);

            if (id != llama_vocab_eos(m_vocab_llama))
            {
                // add it to the context
                m_embd.push_back(id);

                text_to_speak += llama_token_to_piece(m_ctx, id);
            }
        }

        {
            std::string last_output;
            for (int i = m_embd_inp.size() - 16; i < (int)m_embd_inp.size(); i++)
            {
                last_output += llama_token_to_piece(m_ctx, m_embd_inp[i]);
            }
            last_output += llama_token_to_piece(m_ctx, m_embd[0]);

            for (std::string &antiprompt : m_antiprompts)
            {
                if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos)
                {
                    done = true;
                    text_to_speak = ::replace(text_to_speak, antiprompt, "");
                    fflush(stdout);
                    m_need_to_save_session = true;
                    break;
                }
            }
        }

        {
            std::string last_output;
            for (int i = m_embd_inp.size() - 16; i < (int)m_embd_inp.size(); i++)
            {
                last_output += llama_token_to_piece(m_ctx, m_embd_inp[i]);
            }
            last_output += llama_token_to_piece(m_ctx, m_embd[0]);

            for (std::string &antiprompt : m_antiprompts)
            {
                if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos)
                {
                    done = true;
                    text_to_speak = ::replace(text_to_speak, antiprompt, "");
                    fflush(stdout);
                    m_need_to_save_session = true;
                    break;
                }
            }
        }

        m_is_running = sdl_poll_events();

        if (!m_is_running)
        {
            break;
        }
    }
    return text_to_speak;
}