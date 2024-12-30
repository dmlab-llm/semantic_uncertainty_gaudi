import os
import pandas as pd
import torch
import time
import json
import argparse
import struct
import contextlib
from utils import initialize_model, setup_parser, logger, print_stats


def measure_perf(txt="", tps=False, reset=False):
    if hasattr(measure_perf, "prev") and not reset and txt:
        duration = time.perf_counter()-measure_perf.prev
        if tps:
            tps_str = f"Throughput: {(args.batch_size * args.max_new_tokens)/duration:.0f} TPS"
        else:
            tps_str = ""
        logger.info(
            f"{txt} took {duration:.3f} sec. {tps_str}")
    else:
        duration = 0
    measure_perf.prev = time.perf_counter()
    return duration

def get_ds(args):
    ds = pd.read_pickle(args.dataset)

    if args.n_iterations:
        ds = ds.head(args.n_iterations * args.batch_size)

    return ds


def get_input(ds, batch_size):
    queries = []
    tok_input = ds.tolist()
    for start in range(0, len(ds), batch_size):
        end = start + batch_size
        batch = tok_input[start:end]
        input_ids = []
        attention_mask = []
        for query in batch:
            input_ids.append(
                [0] * (args.max_input_tokens - len(query)) + query)
            attention_mask.append(
                [0] * (args.max_input_tokens - len(query)) + [1] * len(query))
        queries.append({
            'input_ids': torch.tensor(input_ids, dtype=torch.int32),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.int32)
        })
    return queries

def main(args, input, prompt):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    print_logs = (local_rank == 0)
    print(f"Dataset has {len(ds)} samples.")
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer, generation_config = initialize_model(args, logger)
    
    data_with_prompts = []
    for text in input:
        data_with_prompts.append(prompt + text + "\n*** ANSWER: ")
    inputs = tokenizer(data_with_prompts, return_tensors="pt").to(args.device)
    results = []
    for input in inputs:
        outputs = model.generate(**input,
                max_new_tokens=4096,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=1.0,
                do_sample=True,
                stopping_criteria=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        if len(outputs.sequences[0]) > 4096:
            outputs.sequences[0] = outputs.sequences[0][:4096]
        full_answer = tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        # For some models, we need to remove the input_data from the answer.
        # print(input_data, full_answer)
        if "* ANSWER: " in full_answer:
            input_data_offset = full_answer.index("* ANSWER: ") + 10
            
        elif full_answer.startswith(input):
            input_data_offset = len(input)
            
        else:
            raise ValueError('Have not tested this in a while.')

        # Remove input from answer.
        answer = full_answer[input_data_offset:]

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()

        # Get the number of tokens until the stop word comes up.
        # Note: Indexing with `stop_at` already excludes the stop_token.
        # Note: It's important we do this with full answer, since there might be
        # non-trivial interactions between the input_data and generated part
        # in tokenization (particularly around whitespaces.)
        token_stop_index = tokenizer(full_answer[:input_data_offset], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(inputs['input_ids'][0])
        n_generated = token_stop_index - n_input_token

        if n_generated == 0:
            n_generated = 1

        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        if len(hidden) == 1:
            last_input = hidden[0]
        elif ((n_generated - 1) >= len(hidden)):
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
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        # Transition_scores[0] only contains the scores for the first generated tokens.

        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == 0:
            raise ValueError

        results.append([sliced_answer, log_likelihoods, last_token_embedding])
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = setup_parser(parser)

    #ds = get_ds(args)
    ds = ["South Korea's Lim Si-hyeon has claimed her third gold medal of the Paris 2024 Olympics having won the women's individual title in archery. The 21-year-old, who beat Great Britain's Megan Havers earlier today in the round of 16, overcame team-mate Nam Su-hyeon 7-3 in the women's final. That completed a clean sweep for Lim as she also won the women's team and mixed team titles. Lisa Barbelin of France claimed a 6-4 win over another South Korean - Jeon Hun-young - to win the bronze medal."]
    prompt = "From the following given text, choose one of the most important sentences that contains the meaning of the whole text and print it as it is."
    main(args, ds, prompt)