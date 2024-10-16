from transformers import pipeline
import os
import torch
import csv
import pandas as pd

from habana_frameworks.torch.hpu import wrap_in_hpu_graph
import habana_frameworks.torch.core as htcore

os.environ["HPU_VISIBLE_DEVICES"] = "6"

model_id = "meta-llama/Meta-Llama-3.1-8B"

generator = pipeline('text-generation', model = model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device="hpu")
generator.model = wrap_in_hpu_graph(generator.model)

path = "/root/datasets/arxiv100.csv"
dataset = pd.read_csv(path)

contexts = []
result = []
prompts = []
for i, data in enumerate(dataset):
    if i < 1:
        text = data["text"]
        contexts.append(text)
        whole_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for information extraction tasks. Please answer in one sentence from the context without any modifications. Keep the answer concise with no explanations. Print out the sentences in the paragraph without modification. Let's think step-by-step.<|eot_id|><|start_header_id|>user<|end_header_id|>
                Question: From the given text, what is the most important sentence that contains the meaning of the whole text? 
                Context: {text} 
                Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """
        prompts.append(whole_text)

#print (f"=== output 0 ===: \n  {output[0][0]['generated_text']} \n")
results = pipeline(prompts)
print(results)