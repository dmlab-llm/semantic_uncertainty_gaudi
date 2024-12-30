import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
#from analyze_results import analyze_run
from uncertainty.data.data_utils import load_ds
from uncertainty.uncertainty_measures.p_ik import get_p_ik
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import context_entails_response
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
import numpy as np
import torch
import gc

 
def eval_semantic_uncertainty(response, logs):
    entailment_model = EntailmentDeberta()

    se = []
    cnt = 0
    for responses, log in zip(tqdm(response), logs):
        r = []
        l = []
        for res in responses:
            log_for_one = []
            if type(res) == str:
                r.append(res)
                for ele in log:
                    log_for_one.append([float(f) for f in ele[1:-1].split(", ")])
                l.append(log_for_one)  
        # print(l)
        semantic_ids = get_semantic_ids(
                        r, model=entailment_model,
                        strict_entailment=True, example=None)
        #log_liks_agg = [0.2 for i in range(5)]

        log_liks_agg = [sum(log_lik)/len(log_lik) for log_lik in log_for_one]
        log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
        pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
        se.append(pe)
    return se


def main():
    dataset = pd.DataFrame()
    for i in range(10):
        part = str(i)
        path = "/root/datasets/arxiv_sent/llama3_"+part+"_temp_1_transition_scores.csv"
        data_part = pd.read_csv(path)
        if i == 0:
            dataset = data_part
        else:
            dataset = pd.concat([dataset, data_part])
    response_index_5 = []
    response_log_index_5 = []
    for i in range(5):
        response_index_5.append(str(i))
        response_log_index_5.append("log_" + str(i))


    response = dataset[response_index_5].values.tolist()
    response_log = dataset[response_log_index_5].values.tolist()
    se = eval_semantic_uncertainty(response, response_log)

    dataset["uncertainty"] = se
    dataset.to_csv("/root/datasets/arxiv_sent/llama3_temp_1_arxiv_semantic_uncertainty.csv")

    _, _, bars = plt.hist(se, alpha = 0.5)
    plt.bar_label(bars)
    plt.savefig("/root/datasets/arxiv_gen/images_llama3_temp_1_arxiv_semantic_uncertainty.png")

    
if __name__ == "__main__":
    main()