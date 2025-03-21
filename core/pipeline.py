
import sys
sys.path.insert(0, "/home/USER/repository/TensorGuard")
from utils.utils import get_pipe, load_json
from utils.constants import MODEL_NAMES
from transformers import pipeline
import torch

import argparse
import json
import copy
import time
import ast
import gc
import os



BASE_PATH = "line_vul_detection_experiment_results"

chat_template = [
    {"role": "system", "content": "You are a security vulnerability expert in C language and good at static program analysis."},
    {"role": "user", "content": "Below is a C code from linux kernel.\n{}\nAnalyze this code and check if there are any vulnerable lines in the code. If you find any vulnerability, output the vulnerable lines or statements as a json compatible list wrapped by ```. If there is no vulnerability return an empty list like ```[]```. Do not add anything other than the list in your response"},
    {"role": "assistant", "content": "```{}```"},
    {"role": "user", "content": "Below is a C code from linux kernel.\n{}\nAnalyze this code and check if there are any vulnerable lines in the code. If you find any vulnerability, output the vulnerable lines or statements as a json compatible list wrapped by ```. If there is no vulnerability return an empty list like ```[]```. Do not add anything other than the list in your response"},
    {"role": "assistant", "content": "```{}```"},
    {"role": "user", "content": "Below is a C code from linux kernel.\n{}\nAnalyze this code and check if there are any vulnerable lines in the code. If you find any vulnerability, output the vulnerable lines or statements as a json compatible list wrapped by ```. If there is no vulnerability return an empty list like ```[]```. Do not add anything other than the list in your response"},
    {"role": "assistant", "content": "```{}```"},
    {"role": "user", "content": "Below is a C code from linux kernel.\n{}\nAnalyze this code and check if there are any vulnerable lines in the code. If you find any vulnerability, output the vulnerable lines or statements as a json compatible list wrapped by ```. If there is no vulnerability return an empty list like ```[]```. Do not add anything other than the list in your response"},
]

def free_mem():
    torch.cuda.empty_cache()
    gc.collect()

def inference(
        patch,
        temperature: int,
        model_id: str):
    
    code_pipe = get_pipe(model_id, temperature=temperature)
    results = []
    for item in data:
        for change in item.get("changes", []):
            for patch in change.get("patches", []):
                input_text = f"Classify the following code patch as either 'buggy' or 'not buggy':\n{patch.get('whole_hunk', '')}"
                print(input_text)
                if input_text:
                    output = code_pipe(input_text)
                    print(output)
                #     results.append({"input": input_text, "output": output})
    
    return results

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="SecVulEval evaluation script.")

    # parser.add_argument('--model_id', type=str, required=True, help='The HuggingFace model name to run on SecVulEval.')
    # parser.add_argument('--temperature', type=float, required=False, default=0.3, help='Model temperature.')

    # args = parser.parse_args()
    
    
    
    data = load_json('data/taxonomy_data/tensorflow_test_data.json') 
    temperature = 0.7
    for item in data:
        for change in item['changes']:
            for patch in change['patches']:
                results = inference(patch['hunk'], temperature, MODEL_NAMES['code_llama_7b'])
                for res in results:
                    print("Input:", res["input"])
                    print("Output:", res["output"])
                    print("="*80)
