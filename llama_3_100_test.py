import copy
import glob
import os
import shutil
import tempfile
import time
from time import time
from pathlib import Path
import logging
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import check_min_version
import argparse
import nltk
from nltk.tokenize import sent_tokenize
from optimum.habana.checkpoint_utils import (
            get_ds_injection_policy,
            get_repo_root,
            model_is_optimized,
            model_on_meta,
            write_checkpoints_json,
            )
from optimum.habana.utils import check_habana_frameworks_min_version, check_optimum_habana_min_version, set_seed
from optimum.habana.transformers.modeling_utils import (
                    adapt_transformers_to_gaudi,
                    )
import multiprocessing



nltk.download('punkt_tab')

parser = argparse.ArgumentParser(description="for multiprocessing llama")
parser.add_argument('part', help='partition number', type = str)
parser.add_argument('gen', help='generation number', type =str)
parser.add_argument('gpu', help='gpu number', type=str)


args = parser.parse_args()
gen = args.gen
part = args.part
gpu = args.gpu
os.environ["HABANA_VISIBLE_DEVICES"]=gpu

class LlamaModel:
    def __init__(self, model_id_or_path: str):
        adapt_transformers_to_gaudi()
        self.device = torch.device("hpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=False)
        hf_config = AutoConfig.from_pretrained(model_id_or_path, torchscript=True, trust_remote_code=False,)

        # Load the model in Gaudi 
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path, config=hf_config, torch_dtype=torch.float32, max_memory={0: '80GIB'}, )
        self.model = model.eval().to(self.device)

        #from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        #self.model = wrap_in_hpu_graph(model)
        self.tokenizer.pad_token_id = self.model.generation_config.pad_token_id
        self.tokenizer.padding_side = "left"

    def tokenize(self, prompt: str):
        """Tokenize the input and move to HPU."""
        input_tokens = self.tokenizer(prompt, return_tensors="pt")
        return input_tokens.input_ids.to(device=self.device)

    def generate(self, prompt: str, **config: dict[str, any]):
        """Take a prompt and generate a response."""
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(device=self.device)
        gen_tokens = self.model.generate(**input_ids, **config, pad_token_id=self.tokenizer.eos_token_id, )
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        
    def predict(self, prompt:str, **config:dict[str,any]):
        """Take a prompt and generate a response."""
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(device=self.device)
        # print(input_ids)
        with torch.no_grad():
            outputs = self.model.generate(**input_ids, **config, return_dict_in_generate=True,
                                output_scores=True,
                                output_hidden_states=True,
                                temperature=1.0,
                                do_sample=True,
                                max_new_tokens=5000,
                                pad_token_id=self.tokenizer.eos_token_id,
                                )
            # print(outputs)
        #if len(outputs.sequences[0]) > 4096: # llama2 max token length
        #	outputs.sequences[0] = outputs.sequences[0][:self.token_limit]
        full_answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        if "*** Answer: " in full_answer:
           input_data_offset = full_answer.index("*** Answer: ") + 12
            
        elif full_answer.startswith(prompt):
            input_data_offset = len(input_ids[0])
            
        else:
            raise ValueError('Have not tested this in a while.')
        answer = full_answer[input_data_offset:]
        stop_at= len(answer) 
        sliced_answer = answer.strip()
#        token_stop_index = len(outputs.sequences[0])
        token_stop_index = self.tokenizer(full_answer[:stop_at + input_data_offset], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(input_ids[0])
        n_generated = token_stop_index - n_input_token

        if n_generated == 0:
            logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
            n_generated = 1
        
        # get last hidden layer
        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        if len(hidden) == 1:
            logging.warning(
                'Taking first and only generation for hidden! '
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer,
                )
            last_input = hidden[0]
        elif ((n_generated - 1) >= len(hidden)):
            # If access idx is larger/equal.
            logging.error(
                'Taking last state because n_generated is too large'
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s, slice_answer: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer, sliced_answer
                )
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        # Then access last layer for input
        last_layer = last_input[-1]
        # Then access last token in input.
        last_token_embedding = last_layer[:, -1, :].cpu()

        # Get log_likelihoods.
        # outputs.scores are the logits for the generated token.
        # outputs.scores is a tuple of len = n_generated_tokens.
        # Each entry is shape (bs, vocabulary size).
        # outputs.sequences is the sequence of all tokens: input and generated.
        transition_scores = self.model.compute_transition_scores(
          	outputs.sequences, outputs.scores, normalize_logits=True)
        # Transition_scores[0] only contains the scores for the first generated tokens.

        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning('Taking first and only generation for log likelihood!')
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        #if len(log_likelihoods) == self.max_new_tokens:
        #    logging.warning('Generation interrupted by max_token limit.')

        if len(log_likelihoods) == 0:
            raise ValueError

        return sliced_answer, log_likelihoods, last_token_embedding
        
model = LlamaModel("meta-llama/Meta-Llama-3.1-8b")

path = "/root/datasets/arxiv/arxiv_cs_" + part
dataset = pd.read_csv(path + ".csv")

short_dataset = pd.DataFrame()
datas = dataset["abstract"]
print("data size: ", len(datas))
prompt = "From the given text, choose one of the most important sentences that contains the meaning of the whole text and print it as it is.\n"
inputs = []
input_len_max = 0

start = time()
## first 100 datas for test
for i, data in enumerate(datas):
    inputs.append(data+ prompt + "*** Answer: ")
print("start generation")
result = []
result_log = []
for i, data in enumerate(inputs):
    predicted_answer, token_log_likelihoods, _ = model.predict(data)
    result.append(predicted_answer)
    result_log.append(token_log_likelihoods)
    if i % 50 == 0:
        print(part, i, time()-start)
end = time()

print("time elapsed: ", end-start)
results = "res_"+gen
reslog = "res_log_"+gen
dataset[results] = result
dataset[reslog] = result_log

dataset.to_csv(path + "_gen_"+gen+".csv")
