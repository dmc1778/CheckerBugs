
import pandas as pd
from collections import Counter
import os, re, json, tiktoken, backoff, csv
from openai import OpenAI
import time, random
import tiktoken
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, model_validator
import sys
sys.path.insert(0, '/home/nima/repository/TensorGuard')
from utils.utils import load_json, write_to_csv, write_list_to_txt, read_txt, separate_added_deleted, is_buggy, file_io
from models.registered_models import LLMs
import logging
from typing import List, Dict
from utils.constants import MODEL_NAMES
from colorama import Fore, Back, Style
import multiprocessing
import concurrent.futures
# load_dotenv()
# client = OpenAI(
#     api_key=os.environ.get("OPENAI_KEY")
# )

model_obj = LLMs()
# logging.basicConfig(level = logging.DEBUG)
# logger = logging.getLogger()

class PromptTemplateStructure(BaseModel):
    Decision: str =  Field(description="YES means the code snippet has a checker bug, NO means the code snippet has no checker bug.")
    Root:  str =  Field(description="The model's reasoning steps")
    
class MyPrompts:
    def __init__(self, shots, strategy: str):
        self.cot_1 = False
        self.cot_2 = False
        self.cot_3 = False
        self.strategy = strategy
        self.shots = shots
        
        if self.strategy == 'cot_1':
            self.cot_1 = True
        elif self.strategy == 'cot_2':
            self.cot_2 = True
        elif self.strategy == 'cot_3':
            self.cot_3 = True
        else:
            return
        
    def setPrompt(self):
        if self.cot_1:
            prompt = """
            You are a software engineer specializing in static analysis and bug detection AI frameworks. 
            Your task is to classify the following code snippet as buggy or not. 
            Follow the steps below to reason through the problem and arrive at a conclusion.
                
            1. Review the Code snippet: Analyze the given code snippet from TensorFlow and PyTorch.
                
            2. Identify potential checker bug violations: Look for any missing, improper, or insufficient, or unnecessary checker statements within the code snippet. Checker statements are generally used to perform error handling, input validation, boundary checking, or other code safety checking.
            
            3. Review code elements: Analyze the code snippet to identify possible code element affected by the violations.
                
            4. Analyze the Impact: Consider the impact of the identified issues on the functionality and reliability of AI frameworks. 

            5. Make a Decision: Based on the above analysis, decide if the code snippet has a checker bug or not.

            6. Generate YES or NO response.
            
            7. Generate your reasoning steps based on above steps, limit your generated tokens to 1000.
            
            Label:
            \n{format_instructions}\n{query}\n"""
            return prompt
        
        elif self.cot_2:
            prompt = """
            You are a software engineer specializing in static analysis and bug detection AI frameworks. 
            Your task is to classify the following code snippet as buggy or not. 
            Follow the steps below to reason through the problem and arrive at a conclusion.
                
            1. Review the Code snippet: Analyze the given code snippet from TensorFlow and PyTorch.
                
            2. Identify potential checker bug violations: Look for any missing, improper, or insufficient, or unnecessary checker statements within the code snippet. Checker statements are generally used to perform error handling, input validation, boundary checking, or other code safety checking.
            
            3. Review code elements: Analyze the code snippet to identify possible code element affected by the violations. Consider the following elements:
                ** Edge cases
                ** Null values
                ** Boundary values
                ** Tensor or variable types
                ** Objects such as device types
                ** Execution modes including eager or graph execution
                ** Quantized tensors
                ** Computation graphs
                
            4. Analyze the Impact: Consider the impact of the identified issues on the functionality and reliability of AI frameworks. 

            5. Make a Decision: Based on the above analysis, decide if the code snippet has a checker bug or not.
            
            6. Generate YES or NO response.
            
            7. Generate your reasoning steps based on above steps, limit your generated tokens to 1000.
            \n{format_instructions}\n{query}\n"""
            return prompt
        
        elif self.cot_3:
            prompt = """
            You are a software engineer specializing in static analysis and bug detection AI frameworks. 
            Your task is to classify the following code snippet as buggy or not. 
            Follow the steps below to reason through the problem and arrive at a conclusion.
            
            Here are some examples buggy code and the fix:
            """
            for i, shot in enumerate(self.shots, start=1):
                deleted_lines = shot["Deleted lines"]
                label = shot["Label"]
                    
            prompt += f"""
            Example {i}:
            - Buggy snippet: "{deleted_lines}"
            Label: {label}
            """
            
            prompt += """
            1. Review the Code snippet: Analyze the given code snippet from TensorFlow and PyTorch.
                
            2. Identify potential checker bug violations: Look for any missing, improper, or insufficient, or unnecessary checker statements within the code snippet. Checker statements are generally used to perform error handling, input validation, boundary checking, or other code safety checking.
            
            3. Review code elements: Analyze the code snippet to identify possible code element affected by the violations.
                
            4. Analyze the Impact: Consider the impact of the identified issues on the functionality and reliability of AI frameworks. 

            5. Make a Decision: Based on the above analysis, decide if the code snippet has a checker bug or not.

            6. Generate YES or NO response.
            
            7. Generate your reasoning steps based on above steps, limit your generated tokens to 1000.
            \n{format_instructions}\n{query}\n"""
            return prompt
        else:
            return

def _chainOp(provider, model_name, temperature, shots, strategy='cot_1'):

    llm = model_obj.get_llm(provider, model_name, temperature)
    
    prompt_obj = MyPrompts(shots, strategy)
    parser = PydanticOutputParser(pydantic_object=PromptTemplateStructure)
    
    prompt = PromptTemplate(
        template=prompt_obj.setPrompt(),
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    prompt_and_model = prompt | llm
    return prompt_and_model, parser

class ConstructPrompt:
    def __init__(self):
        pass
    
    def get(self, strategy, shots=None):
        prompt = ""
        if strategy == 'zero':
            prompt += self.zero_shot()
        elif strategy == 'few':
            prompt += self.few_shot(shots)
        elif strategy == 'cot_1':
            prompt += self.cot_1()
        elif strategy == 'cot_2':
            prompt += self.cot_2()
        else:
            raise("No prompting strategy is provided.")
        return prompt
    
    def zero_shot(self):
        prompt = """
        You are a software engineer specializing in static analysis and bug detection in AI frameworks. Your task is to classify the following code snippet as buggy or not. Based on your analysis, decide if the code snippet has a checker bug or not. Please generate YES or NO response without any further explanation. <CODE SNIPPET START FROM HERE:>
        """
        return prompt
        
    def few_shot(self, shots):
        prompt = """
        You are a software engineer specializing in static analysis and bug detection AI frameworks. Your task is to classify the following code snippet as buggy or not. Look for any missing, improper, or insufficient checker statements within the code snippet. Checker statements are generally used to perform error handling, input validation, boundary checking, or other code safety checking. Based on your analysis, decide if the code snippet has a checker bug or not. 

        Here are some examples buggy code and the fix:
        """
        for i, shot in enumerate(shots, start=1):
            deleted_lines = shot["Deleted lines"]
            added_lines = shot["Added lines"]
            label = shot["Label"]
            commit_message = shot["Commit message"]
                
        prompt += f"""
        Example {i}:
        - Buggy snippet: "{deleted_lines}"
        + Fixed snippet: "{added_lines}"
        Label: {label}
        Commit message: "{commit_message}"
        """

        prompt += """
        Please generate YES or NO response without any further explanation.
        
        <CODE SNIPPET START FROM HERE:>
        """

        return prompt
        
    def cot_1(self):
        prompt = """
            You are a software engineer specializing in static analysis and bug detection AI frameworks. 
            Your task is to classify the following code snippet as buggy or not. 
            Follow the steps below to reason through the problem and arrive at a conclusion.
                
            1. Review the Code snippet: Analyze the given code snippet from TensorFlow and PyTorch.
                
            2. Identify potential checker bug violations: Look for any missing, improper, or insufficient, or unnecessary checker statements within the code snippet. Checker statements are generally used to perform error handling, input validation, boundary checking, or other code safety checking.
            
            3. Review code elements: Analyze the code snippet to identify possible code element affected by the violations.
                
            4. Analyze the Impact: Consider the impact of the identified issues on the functionality and reliability of AI frameworks. 

            5. Make a Decision: Based on the above analysis, decide if the code snippet has a checker bug or not.

            6. Please generate YES or NO response without any further explanation.
                
            <CODE SNIPPET START FROM HERE:> 
        """
        return prompt
    
    def cot_2(self):
        prompt = """
            You are a software engineer specializing in static analysis and bug detection AI frameworks. 
            Your task is to classify the following code snippet as buggy or not. 
            Follow the steps below to reason through the problem and arrive at a conclusion.
                
            1. Review the Code snippet: Analyze the given code snippet from TensorFlow and PyTorch.
                
            2. Identify potential checker bug violations: Look for any missing, improper, or insufficient, or unnecessary checker statements within the code snippet. Checker statements are generally used to perform error handling, input validation, boundary checking, or other code safety checking.
            
            3. Review code elements: Analyze the code snippet to identify possible code element affected by the violations. Consider the following elements:
                ** Edge cases
                ** Null values
                ** Boundary values
                ** Tensor or variable types
                ** Objects such as device types
                ** Execution modes including eager or graph execution
                ** Quantized tensors
                ** Computation graphs
                
            4. Analyze the Impact: Consider the impact of the identified issues on the functionality and reliability of AI frameworks. 

            5. Make a Decision: Based on the above analysis, decide if the code snippet has a checker bug or not.

            6. Please generate YES or NO response without any further explanation.
                
            <CODE SNIPPET START FROM HERE:> 
        """
        return prompt

# 5. Please generate YES or NO response and explain root cause behind your decision.

def process_instance(args):
    """Worker function for processing a single instance."""
    iteration, lib_name, model_name, temperature, strategy, instance, rule_data, prompt_obj, idx, data, provider, mode = args
    # log_path = f'logs/{lib_name}/{iteration}/{model_name}::{temperature}::{strategy}/log.txt'
    # hist = read_txt(log_path)

    # if instance['commit_link'] in hist:
    #     print(Fore.YELLOW + "This instance has been already processed.")
    #     return

    # write_list_to_txt(instance['commit_link'], log_path)
    
    if strategy == 'cot_3':
        rand_num = random.randint(1, 13)
        _shot = [rule_data[f"entry{rand_num}"]['example1'], rule_data[f"entry{rand_num}"]['example2']]
        if instance['commit_link'] in (_shot[0]['commit_link'], _shot[1]['commit_link']):
            print(Fore.RED + "This instance is among one of the shots, skipping!")
            return

    #     prompt_template = prompt_obj.get(strategy, _shot)
    # else:
    #     prompt_template = prompt_obj.get(strategy)

    for change in instance['changes']:
        if not change or 'test' in change['path'] or 'tests' in change['path']:
            continue

        for k, patch in enumerate(change['patches']):
            if not patch:
                continue    
            
            try:
                #llm = model_obj.get_llm(provider, model_name, temperature)
                if provider == 'fireworks':
                    time.sleep(3)
                elif provider == 'openai':
                    time.sleep(2)
                else:
                    time.sleep(6)
                    
                code_snippet = patch['hunk_buggy'].replace('-', '')
                #output = llm.invoke(f"{prompt_template} {code_snippet}")
                
                prompt_and_model, parser = _chainOp(provider, model_name, temperature, _shot, strategy=strategy)
                
                output = prompt_and_model.invoke({"query": code_snippet})
                
                structured_output = parser.invoke(output)
                
                # result_dict = structured_output.dict()
                
                output_row = [iteration, lib_name, model_name, temperature, strategy, instance['commit_link'], change['path'], True, structured_output.Decision, structured_output.Root]

                
                if provider == 'fireworks':
                    model_name = model_name.split('/')[3]
                    write_to_csv(output_row, lib_name, iteration, model_name, mode)
                else:
                    write_to_csv(output_row, lib_name, iteration, model_name, mode)

                print(Fore.CYAN + f"Iteration: {iteration} | Provider: {provider} | Model: {model_name} | Temp: {temperature} | Strategy: {strategy} | Instance: {idx}/{len(data)} | Patch Number: {k+1} | Decision: {output.content}" + Style.RESET_ALL)
            except Exception as e:
                if re.findall(r"(\"Decision\"\:\s\"NO\")", output.content):
                    output_row = [iteration, lib_name, model_name, temperature, strategy, instance['commit_link'], change['path'], True, 'NO', structured_output.Root]
                elif re.findall(r"(\"Decision\"\:\s\"YES\")", output.content):
                    output_row = [iteration, lib_name, model_name, temperature, strategy, instance['commit_link'], change['path'], True, 'YES', output.content]
                else:
                    output_row = [iteration, lib_name, model_name, temperature, strategy, instance['commit_link'], change['path'], True, 'Parse Error', output.content]
                
                if provider == 'fireworks':
                    model_name = model_name.split('/')[3]
                    write_to_csv(output_row, lib_name, iteration, model_name, mode)
                else:
                    write_to_csv(output_row, lib_name, iteration, model_name, mode)
                print(Fore.RED + str(e))
                time.sleep(10)

def main(args):
    lib_name = args[0]
    num_iter = args[1]
    providers = args[2]
    mode = args[3]
    
    data_path = f"data/taxonomy_data/{lib_name}_test_data.json"
    rule_path = "data/rule_set.json"

    rule_data = load_json(rule_path)
    data = load_json(data_path)

    # prompt_obj = ConstructPrompt()

    for iteration in range(1, num_iter):
        tasks = []
        for provider in [providers]:
            current_models = MODEL_NAMES[provider]

        for model_name in current_models:
                # i / 10 for i in range(7, 8)
            for temperature in [0.6]:
                for strategy in ['cot_3']:
                    for idx, instance in enumerate(data):
                            # process_instance(iteration, lib_name, model_name, temperature, strategy, instance, rule_data, prompt_obj, idx, data, provider, mode)
                        tasks.append((iteration, lib_name, model_name, temperature, strategy, instance, rule_data, None, idx, data, provider, mode))
                
        with multiprocessing.Pool(24) as pool:
            try:
                for _ in pool.imap_unordered(process_instance, tasks):
                    pass
            finally:
                pool.close()
                pool.join()
                pool.terminate()
    


if __name__ == "__main__":
    libname = sys.argv[1]
    num_iter = sys.argv[2]
    providers = sys.argv[3]
    mode = sys.argv[4]

    # libname = 'pytorch'
    # num_iter = 2
    # providers = 'fireworks'
    # mode = 'regular'
    
    args = [libname, int(num_iter), providers, mode]
    main(args)
    
