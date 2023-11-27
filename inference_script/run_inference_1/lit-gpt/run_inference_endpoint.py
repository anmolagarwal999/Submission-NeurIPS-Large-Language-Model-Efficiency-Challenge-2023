# script originally originated from: https://github.com/nlpxucan/WizardLM/blob/main/WizardCoder/src/humaneval_gen.py
import argparse
import pprint
import sys
import os
os.environ["HUGGINGFACE_HUB_CACHE"]="/submission/hf_cache"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
import re, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
#####
from transformers import LlamaForCausalLM, CodeLlamaTokenizer
from transformers import LlamaForCausalLM
import colorama
colorama.init()
####

####
import torch
from huggingface_hub import login
import time
from optimum.bettertransformer import BetterTransformer
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
import datetime
import pytz
import subprocess
import pynvml
pynvml.nvmlInit()
###################################
#### IMPORT THE UTILS
import eval_related_utils.eval_helper as eval_helper
# importlib.reload(eval_helper)
from eval_related_utils.eval_helper import find_category
from eval_related_utils.eval_helper import QueryCategory
from eval_related_utils.eval_helper import FetchModel


#########################################
from fastapi import FastAPI

import logging

# Lit-GPT imports
import sys
import time
from pathlib import Path
import json

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lightning as L
import torch

torch.set_float32_matmul_precision("high")

from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token
)

app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

#######################
# LLAMA-2 related inference code can be verified from here: https://github.com/facebookresearch/llama-recipes/blob/main/inference/inference.py
##################
login(token="<insert huggingface token>")
print("Huggingface login done")
# sys.exit(0)
start = time.time()
print(f"Starting time is: {start}")


##################
where_to_eval_map = {
    QueryCategory.bbq_query: "<Huggingface model string to model 1 of the ensemble | Load checkpoint corresponding to 243rd step of the 2nd epoch>",
    QueryCategory.tqa_query: "<Huggingface model string to model 2 of the ensemble | Load checkpoint corresponding to 243rd step of the 3rd epoch>",    
}
where_to_eval_map[QueryCategory.mmlu_query] = where_to_eval_map[QueryCategory.bbq_query]
where_to_eval_map[QueryCategory.cnn_query] = where_to_eval_map[QueryCategory.bbq_query]
where_to_eval_map[QueryCategory.gsm_category] = where_to_eval_map[QueryCategory.bbq_query]
where_to_eval_map[QueryCategory.bigbench_category] = where_to_eval_map[QueryCategory.bbq_query]
where_to_eval_map[QueryCategory.other_category] = where_to_eval_map[QueryCategory.tqa_query]
###################
fetch_model_obj = FetchModel(where_to_eval_map)

fetch_model_obj.print_model_statuses()


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    
    print(colorama.Fore.RED)
    print("Input is: ")
    print(json.dumps(json.loads(input_data.json()), indent=2))
    print(colorama.Fore.RESET)
    ##################
    # model_cat = find_category()
    # print("Category chosen is: ", model_cat)
    model, tokenizer = fetch_model_obj.fetch_model_tokenizer_for_query(input_data.prompt)
    
    ################
    
    # RESET INPUT
    # earlier_prompt = input_data.prompt
    # input_data.prompt=  invert_mildmix(input_data.prompt)
    # if earlier_prompt!=input_data.prompt:
    #     print("PROMPT has been changed")
    
    ###################
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)
    else:
        torch.manual_seed(42)
    
    # contains input_ids and attention masks
    encoded = tokenizer(input_data.prompt, return_tensors="pt")
    
    # stores the input length of the prompt
    prompt_length = encoded["input_ids"][0].size(0)
    
    # find the maximum number of tokens that can be returned
    max_num_tokens_returnable = prompt_length + input_data.max_new_tokens
    # assert max_num_tokens_returnable <= LLAMA2_CONTEXT_LENGTH, (
    #     max_num_tokens_returnable,
    #     LLAMA2_CONTEXT_LENGTH,
    # )

    t0 = time.perf_counter()
    
    # convert the input to cuda
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=input_data.max_new_tokens,
            do_sample=True,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    t = time.perf_counter() - t0
    
    # fetch whether the output string should contain only the NEWLY generated tokens or SHOULD IT INCLUDE THE INPUT PROMPT as well
    if not input_data.echo_prompt:
        output_str = tokenizer.decode(outputs.sequences[0][prompt_length:], skip_special_tokens=True)
    else:
        output_str = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
    _num_new_tokens_generated = outputs.sequences[0].size(0) - prompt_length
    logger.info(
        f"Time for inference: {t:.02f} sec total, {_num_new_tokens_generated / t:.02f} tokens/sec"
    )

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    
    # convert the logit scores -> probability scores -> log-likelihood scores
    # shape: torch.Size([1, _num_new_generated_tokens, 32001])
    log_probs = torch.log(torch.stack(outputs.scores, dim=1).softmax(-1))

    # Retain the INPUT IDs (ie token IDs) of only the newly generated tokens
    # torch.Size([1, _num_new_tokens_generated]) | torch.int64
    gen_sequences = outputs.sequences[   :     ,     encoded["input_ids"].shape[-1]:    ]
    
    # fetches the log likelihood corresponding to all the newly generated tokens for the CHOSEN TOKEN ID only
    # torch.Size([1, _num_new_tokens_generated]) | torch.float32
    gen_logprobs = torch.gather(log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)

    # fetch the top-ranked token as per log-likelihood scores for each newly generated position
    # torch.Size([1, _num_new_tokens_generated])
    top_indices = torch.argmax(log_probs, dim=-1)
    
    # fetches the log likelihood corresponding to all the newly generated tokens for the TOP TOKEN ID
    # torch.Size([1, _num_new_tokens_generated])
    top_logprobs = torch.gather(log_probs, 2, top_indices[:,:,None]).squeeze(-1)
    
    
    top_indices = top_indices.tolist()[0]
    top_logprobs = top_logprobs.tolist()[0]

    generated_tokens = []
    for t, lp, tlp in zip(gen_sequences.tolist()[0], gen_logprobs.tolist()[0], zip(top_indices, top_logprobs)):
        # t  = token ID of the newly generated token at that position
        # lp  = log likelihood corresponding to the newly generated token
        # tlp -> (token ID of the token with the highest log-lieklihood scores in that position, corresponding highest log likelihood scores)
        
        top_token_idx, top_token_log_likelihood = tlp
        top_tok_str = tokenizer.decode(top_token_idx)
        token_tlp = {top_tok_str: top_token_log_likelihood}
        
        generated_tokens.append(
            Token(text=tokenizer.decode(t), # token string corresponding to the token actually outputted
                  logprob=lp,  # log likelihood of the outputted token
                  top_logprob=token_tlp # details about the top-ranked token
                  )
        )
    
    # fetches the sum of log likelihood for the newly generated tokens ONLY
    logprob_sum = gen_logprobs.sum().item()
    
    _output_obj =  ProcessResponse(
        text=output_str, tokens=generated_tokens, logprob=logprob_sum, request_time=t
    )
    ##############
    print(colorama.Fore.GREEN)
    print("OUTPUT is: ")
    print(json.dumps(json.loads(_output_obj.json()), indent=2))
    print(colorama.Fore.RESET)
    
    return _output_obj


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    
    _model, tokenizer = fetch_model_obj.fetch_model_tokenizer_for_query(input_data.text)
    
    t0 = time.perf_counter()
    encoded = tokenizer(
        input_data.text
    )
    t = time.perf_counter() - t0
    tokens = encoded["input_ids"]
    return TokenizeResponse(tokens=tokens, request_time=t)
