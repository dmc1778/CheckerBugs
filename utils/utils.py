
import json
import tiktoken
import csv
import os
from openai import OpenAI

# client = OpenAI(
#     api_key=os.environ.get("OPENAI_KEY")
# )

def file_io(iteration, libname, model_name, temperature, strategy, memorize=False, commit_link=None):
    if memorize:
        write_list_to_txt(commit_link, f'logs/{libname}/{iteration}/{model_name}::{temperature}::{strategy}.txt')
        return None
    else:
        hisotry_file = f'logs/{libname}/{iteration}/{model_name}::{temperature}::{strategy}.txt'
        f1 = open(hisotry_file, 'a')
        return read_txt(f'logs/{libname}/{iteration}/{model_name}::{temperature}::{strategy}.txt')

def get_token_count(string):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def separate_added_deleted(github_diff):
    diff_lines = github_diff.split('\n')

    added_lines = ""
    deleted_lines = ""

    for line in diff_lines:
        if line.startswith('+'):
            added_lines += line[0:] + '\n'
        elif line.startswith('-'):
            deleted_lines += line[0:] + '\n'
    return deleted_lines, added_lines

def read_txt(fname):
    with open(fname, "r") as fileReader:
        data = fileReader.read().splitlines()
    return data

def write_list_to_txt(data, filename):
    with open(filename, "a", encoding='utf-8') as file:
        file.write(data+'\n')

def is_buggy(input_string):
    yes_variants = {"YES", "yes", "Yes"}
    return input_string in yes_variants

def filter_dataset(dataset):
    filtered_dataset = []
    for item in dataset:
        if item['Root Cause'] != 'Others' or item['Root Cause'] != 'others':
            filtered_dataset.append(item)
    return filtered_dataset

def load_json(data_path):
    with open(data_path) as json_file:
        data = json.load(json_file)
    return data

def write_to_csv(data, libname, iteration, model_name, mode=None):
    
    if mode == 'rootcause':
        if not os.path.exists(f"output/rootcause"):
            os.makedirs(f"output/rootcause")
        
        with open(f"output/rootcause/{libname}_{iteration}_{model_name}_results.csv", 'a', encoding="utf-8", newline='\n') as file_writer:
            write = csv.writer(file_writer)
            write.writerow(data)
    else:
        if not os.path.exists(f"output"):
            os.makedirs(f"output")
        
        with open(f"output/{libname}_{iteration}_{model_name}_results.csv", 'a', encoding="utf-8", newline='\n') as file_writer:
            write = csv.writer(file_writer)
            write.writerow(data)


# def completions_with_backoff(prompt, temperature,  model='gpt-3.5-turbo'):
#     response = client.chat.completions.create(
#         model=model,
#         temperature=temperature,
#         messages=[
#             {"role": "system", "content": prompt}
#         ]
#     )
#     return response