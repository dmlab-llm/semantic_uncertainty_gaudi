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
from tqdm import tqdm
import numpy as np
from scipy.special import log_softmax

# os.environ["PT_HPU_LAZY_MODE"]="0"
class LlamaModel:

    def __init__(self, model_id_or_path: str):
        adapt_transformers_to_gaudi()
        self.device = torch.device("hpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=False)
        hf_config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=False,) #torchscript=True, 
        
        # Load the model in Gaudi 
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path, config=hf_config, torch_dtype=torch.bfloat16)
        #from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        #self.model = wrap_in_hpu_graph(model)
        self.model = model.eval().to(self.device)

        
        self.tokenizer.pad_token_id = self.model.generation_config.pad_token_id
        self.tokenizer.padding_side = "left"
        self.max_new_tokens = 4096

    def tokenize(self, prompt: str):
        """Tokenize the input and move to HPU."""
        input_tokens = self.tokenizer(prompt, return_tensors="pt")
        return input_tokens.input_ids.to(device=self.device)

    def generate(self, input_data: str, temperature): #, args
        inputs = self.tokenizer(input_data, return_tensors="pt").to(self.device)
        generation_config = copy.deepcopy(self.model.generation_config)
        # generation_config.use_cache = args.use_kv_cache
        # generation_config.bucket_size = args.bucket_size
        # generation_config.bucket_internal = args.bucket_internal
        #generation_config.trim_logits = args.trim_logits
        #generation_config.attn_softmax_bf16 = args.attn_softmax_bf16
        #generation_config.use_flash_attention = args.use_flash_attention
        #generation_config.flash_attention_recompute = args.flash_attention_recompute
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                max_new_tokens= 4096, #self.max_new_tokens, #1024
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True,
                stopping_criteria=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=False)

#                top_k = 50,
#                eos_token_id=terminators,
#                stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        
        if "<|start_header_id|>assistant<|end_header_id|>" in full_answer:
            input_data_offset = full_answer.index("<|start_header_id|>assistant<|end_header_id|>") + len("<|start_header_id|>assistant<|end_header_id|>")
        else:
            input_data_offset = len(input_data)
        if "<|eom_id|>" in full_answer[input_data_offset:]:
            end = full_answer[input_data_offset:].index("<|eom_id|>")
        elif "<|eot_id|>" in full_answer[input_data_offset:]:
            end = full_answer[input_data_offset:].index("<|eot_id|>")
        else:
            end = len(full_answer[input_data_offset:])
        stop = input_data_offset + end
        answer = full_answer[input_data_offset:stop]
        sliced_answer = answer.strip()
        stop_at = end
        token_stop_index = self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(inputs['input_ids'][0])
        n_generated = token_stop_index - n_input_token

        if n_generated == 0:
            logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
            n_generated = 1

        # Get the last hidden state (last layer) and the last token's embedding of the answer.
        # Note: We do not want this to be the stop token.

        # outputs.hidden_state is a tuple of len = n_generated_tokens.
        # The first hidden state is for the input tokens and is of shape
        #     (n_layers) x (batch_size, input_size, hidden_size).
        # (Note this includes the first generated token!)
        # The remaining hidden states are for the remaining generated tokens and is of shape
        #    (n_layers) x (batch_size, 1, hidden_size).

        # Note: The output embeddings have the shape (batch_size, generated_length, hidden_size).
        # We do not get embeddings for input_data! We thus subtract the n_tokens_in_input from
        # token_stop_index to arrive at the right output.

        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states
        # print(hidden)
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
        #transition_scores = self.model.compute_transition_scores(
        #    outputs.sequences, outputs.scores, normalize_logits=True)
        #print(transition_scores)

        transition_scores1 = self.compute_transition_scores(
            outputs.sequences, outputs.scores, input_data_offset, normalize_logits=True)
        # print(transition_scores1)
        # Transition_scores[0] only contains the scores for the first generated tokens.
        # print("scores: ", transition_scores)
        log_likelihoods = [score.item() for score in transition_scores1]
        if len(log_likelihoods) == 1:
            logging.warning('Taking first and only generation for log likelihood!')
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning('Generation interrupted by max_token limit.')

        if len(log_likelihoods) == 0:
            raise ValueError 

        #probs = torch.log_softmax(outputs.logits, dim=-1).detach()
        #probs = probs[:, :-1, :]
        #generated_input_ids_shifted = inputs[:, 1:]
        #gen_probs = torch.gather(probs, 2, generated_input_ids_shifted[:, :, None]).squeeze(-1)
        #print(gen_probs[:,-self.max_new_tokens:])
        # print(sliced_answer)
        return sliced_answer, log_likelihoods, last_token_embedding
        
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
        # print(full_answer)
        if "[/INST]" in full_answer:
           input_data_offset = full_answer.index("[/INST]") + 7

        else:
            print(full_answer)
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
            #logging.error(
            #    'Taking last state because n_generated is too large'
            #    'n_generated: %d, n_input_token: %d, token_stop_index %d, '
            #    'last_token: %s, generation was: %s, slice_answer: %s',
            #    n_generated, n_input_token, token_stop_index,
            #    self.tokenizer.decode(outputs['sequences'][0][-1]),
            #    full_answer, sliced_answer
            #    )
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
    
    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: tuple[torch.Tensor],
        input_data_offset: int,
        beam_indices = None,
        normalize_logits: bool = False,
    ) -> torch.Tensor:
        if beam_indices is None:
            beam_indices = torch.arange(scores[0].shape[0]).view(-1, 1).to(sequences.device)
            beam_indices = beam_indices.expand(-1, len(scores))
        scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)

        if normalize_logits:
            scores = scores.reshape(-1, self.model.config.vocab_size, scores.shape[-1])
            scores = torch.nn.functional.log_softmax(scores, dim=1)
            scores = scores.reshape(-1, scores.shape[-1])

        beam_indices_mask = beam_indices < 0
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()

        beam_indices = beam_indices.clone()[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]

        beam_indices[beam_indices_mask] = 0
        beam_sequence_indices = beam_indices * self.model.config.vocab_size
        cut_idx = sequences.shape[-1] - max_beam_length 

        for i,seq in enumerate(sequences[0]):
            if seq == 128009 and sequences[0][i+1] == 128009 and sequences[0][i+2] == 128009:
                cut_idx = i - max_beam_length+1
                break
        indices = sequences[:, cut_idx:cut_idx + max_beam_length] + beam_sequence_indices
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append(scores[idx][i])
        transition_scores = torch.stack(results)

        return transition_scores
    
def main():
    parser = argparse.ArgumentParser(description="for llama")
    parser.add_argument('gpu', help='gpu number', type=str)
    parser.add_argument('part', help='part number', type=str)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_kv_cache",action="store_true")
    parser.add_argument("--trim_logits",action="store_true")
    parser.add_argument("--attn_softmax_bf16", action="store_true")
    parser.add_argument(
        "--bucket_size",
        default=-1,
        type=int,
        help="Bucket size to maintain static shapes. If this number is negative (default is -1) \
            then we use `shape = prompt_length + max_new_tokens`. If a positive number is passed \
            we increase the bucket in steps of `bucket_size` instead of allocating to max (`prompt_length + max_new_tokens`).",
    )
    parser.add_argument(
        "--bucket_internal",
        action="store_true",
        help="Split kv sequence into buckets in decode phase. It improves throughput when max_new_tokens is large.",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Whether to enable Habana Flash Attention, provided that the model supports it.",
    )
    parser.add_argument(
        "--flash_attention_recompute",
        action="store_true",
        help="Whether to enable Habana Flash Attention in recompute mode on first token generation. This gives an opportunity of splitting graph internally which helps reduce memory consumption.",
    )

    args = parser.parse_args()
    part = args.part
    gpu = args.gpu
    os.environ["HABANA_VISIBLE_DEVICES"]=gpu

    model = LlamaModel("meta-llama/Meta-Llama-3.1-8B-Instruct")

    path = "/root/datasets/arxiv/arxiv_cs_"

    dataset = pd.read_csv(path + part + ".csv")

    datas = dataset["abstract"]

    inputs = []
    input_len_max = 0

    start = time()
    for data in datas:
        summary_messages = [
            {"role": "system", "content": "You are an assistant for text summarization tasks. Summarize the given text in one sentence containing the meaning of the whole text. The summary has to be shorter than 50 words. Just write the answer. Keep the answer concise."},
            {"role": "user", "content": data.replace("\n", " ")}
        ]
        messages=[
                {"role": "system", "content": "You are an assistant for information extraction tasks. From the given text, what is the most important sentence that contains the meaning of the whole text? Just write the answer. Just print one sentence from the context without any modifications or additional words. Keep the answer concise with no explanations. Print out the sentences in the paragraph as it is with no modification."},
                {"role": "user", "content": data.replace("\n", " ")}
            ]
        prompt = model.tokenizer.apply_chat_template(messages, tokenize=False)
        # print(prompt)
        inputs.append(prompt + "<|start_header_id|>assistant<|end_header_id|>")

    answers = []
    logs = []

    for data in tqdm(inputs):
        result = []
        result_log = []
        for i in range(5):
            predicted_answer, token_log_likelihoods, _ = model.generate(data, 1.0)
            result.append(predicted_answer)
            result_log.append(token_log_likelihoods)
        answers.append(result)
        print(result)
        logs.append(result_log)
    end = time()
    print("time elapsed: ", end-start)
    
    new_dataset = pd.DataFrame()
    new_dataset["abstract"] = datas
    
    ans1 = []
    ans2 = []
    ans3=[]
    ans4=[]
    ans5=[]
    log1 = []
    log2 = []
    log3 = []
    log4 = []
    log5 = []
    for anss, log in zip(answers, logs):
        ans = anss
        l = log
        ans1.append(ans[0])
        ans2.append(ans[1])
        ans3.append(ans[2])
        ans4.append(ans[3])
        ans5.append(ans[4])
        log1.append(l[0])
        log2.append(l[1])
        log3.append(l[2])
        log4.append(l[3])
        log5.append(l[4])

    new_dataset["0"] = ans1
    new_dataset["1"] = ans2
    new_dataset["2"] = ans3
    new_dataset["3"] = ans4
    new_dataset["4"] = ans5

    new_dataset["log_0"] = log1
    new_dataset["log_1"] = log2
    new_dataset["log_2"] = log3
    new_dataset["log_3"] = log4
    new_dataset["log_4"] = log5
#    new_dataset.to_csv("/root/datasets/arxiv_gen/llama3_arxiv_"+"noisy"+"_transition_scores.csv")

    new_dataset.to_csv("/root/datasets/arxiv_sent/llama3_"+ part +"_temp_1_transition_scores.csv")

if __name__ == "__main__":
    main()