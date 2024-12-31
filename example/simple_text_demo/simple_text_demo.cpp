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
#include <iostream>

int main(int argc, char *argv[]) {
    /* Create new model params struct with for the model settings */
    simplellama_model_params model_params;
    model_params.model_llama = "phi-2.Q5_0.gguf"; /* Model name, downloaded automatically by cmake */

    /* Make a new instance of SimpleLLama */
    SimpleLLama sl(model_params);
    
    /* Initialize the model runtime */
    sl.init();
    
    /* Some example questions that we want to get answered :) */
    std::string questions[] = {"What came first, the egg or the chicken?", 
                               "What is the capital city of Germany?", 
                               "What is the capital city of the United States?",
                               "What is 10*100=?"};

    /* Loop through the questions and ask our LLM */
    for(auto question : questions) {
        std::cout << question << '\n';
        std::string response = "response:" + sl.do_inference(question);
        std::cout << response << '\n';
    }
    
    return 0;
}