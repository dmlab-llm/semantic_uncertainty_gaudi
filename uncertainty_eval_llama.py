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

 
def eval_semantic_uncertainty(response, logs, abstracts, prompt):
    #entailment_model = EntailmentDeberta()
    entailment_model = EntailmentLlama(None, False, "meta-llama/Meta-Llama-3.1-8B-Instruct")
    se = []
    num_clusters = []
    log_liks_aggs = []
    sids = []
    log_likelihood_per_semantic_ids= []
    # print(logs)
    for responses, log, abstract in zip(tqdm(response), logs, abstracts):
        #print(responses, log, abstract)
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
                        strict_entailment=True, example=prompt + abstract)
        #log_liks_agg = [0.2 for i in range(5)]
        # print(l)
        log_liks_agg = [sum(log_lik)/len(log_lik) for log_lik in log_for_one]
        log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
        pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
        se.append(pe)
        num_clusters.append(len(set(semantic_ids)))
        print("\nnumber of clusters: ", len(set(semantic_ids)), semantic_ids)
        print(log_liks_agg, log_likelihood_per_semantic_id, pe)
        log_liks_aggs.append(log_liks_agg)
        log_likelihood_per_semantic_ids.append(log_likelihood_per_semantic_id)
        sids.append(semantic_ids)
    return se, num_clusters, log_liks_aggs, log_likelihood_per_semantic_ids, sids


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

    abstract = dataset['abstract']
    keysent_prompt = "You are an assistant for information extraction tasks. From the given text, what is the most important sentence that contains the meaning of the whole text? Just write the answer. Just print one sentence from the context without any modifications or additional words. Keep the answer concise with no explanations. Print out the sentences in the paragraph as it is with no modification."
    prompt = "You are an assistant for text summarization tasks. Summarize the given text in one sentence containing the meaning of the whole text. The summary has to be shorter than 50 words. Just write the answer. Keep the answer concise."
        
    se, num_clusters, log_liks_agg, log_likelihood_per_semantic_id, semantic_ids = eval_semantic_uncertainty(response, response_log, abstract, keysent_prompt)

    dataset["uncertainty"] = se
    dataset['semantic_ids'] = semantic_ids
    dataset['num_clusters'] = num_clusters

    
    plt.subplot(211)
    plt.plot(range(len(se)), sorted(se))
    plt.legend()
    plt.xlabel("data index")
    plt.ylabel("semantic uncertainty")
    
    plt.subplot(212)
    plt.plot(num_clusters, se)
    plt.legend()
    plt.xlabel("number of clusters")
    plt.ylabel("semantic uncertainty")
    plt.savefig("/root/datasets/arxiv_sent/llama3_new_sent_noisy_temp_1_comp_semantic_uncertainty_cluster_llama.png")

    plt.figure(figsize=(10, 20))
    plt.subplot(211)
    for j in log_liks_agg:
        plt.plot(range(len(j)), j, colors[i], alpha = 0.5, ms = 3)

    plt.legend()
    plt.xlabel("data generation tokens")
    plt.ylabel("log_liks_agg")

    plt.subplot(212)
    plt.scatter(num_clusters, se)
    plt.legend()
    plt.xlabel("data generation tokens")
    plt.ylabel("log_likelihood_per_semantic_id")

    plt.savefig("/root/datasets/arxiv_sent/llama3_sent_temp_1_comp_semantic_uncertainty_cluster_llama_2.png")

    dataset.to_csv("/root/datasets/arxiv_sent/llama3_sent_temp_1_arxiv_semantic_uncertainty_cluster_llama.csv")


if __name__ == "__main__":
    main()