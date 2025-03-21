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
sys.path.insert(0, '/home/USER/repository/TensorGuard')
from utils.utils import load_json, write_to_csv, write_list_to_txt, read_txt, separate_added_deleted, is_buggy, file_io
from models.registered_models import LLMs
import logging
from typing import List, Dict
from utils.constants import MODEL_NAMES
from colorama import Fore, Back, Style

# load_dotenv()
# client = OpenAI(
#     api_key=os.environ.get("OPENAI_KEY")
# )

model_obj = LLMs()
# logging.basicConfig(level = logging.DEBUG)
# logger = logging.getLogger()

class PromptTemplateStructure(BaseModel):
    YES: str =  Field(description="YES means the code has a checker bug.")
    NO:  str =  Field(description="NO means the code has no checker bug.")
    
class MyPrompts:
    def __init__(self, strategy: str):
        self.zeroshot_prompt = False
        self.fewshot_prompt = False
        self.cot_prompt = False
        self.strategy = strategy
        
        if self.strategy == 'zero':
            self.zeroshot_prompt = True
        elif self.strategy == 'few':
            self.fewshot_prompt = True
        elif self.strategy == 'cot':
            self.cot_prompt = True
        else:
            return
        
    def setPrompt(self):

        if self.zeroshot_prompt:
            zeroshot_prompt = "You are a software engineer specializing in static analysis and bug detection. Your task is to classify the following code hunk as buggy or not. A checker in C++ refers to a piece of code that verifies conditions, assertions, or constraints to detect software bugs. Based on your analysis, decide if the code hunk is buggy or not. Generate YES or NO response without any further explanations.\n{format_instructions}\n{query}\n"
            return zeroshot_prompt
        
        elif self.fewshot_prompt:
            base_prompt_p2 = " \n{format_instructions}\n{query}\n"
            return base_prompt_p2
        
        elif self.cot_prompt:
            chain_of_thought_ = """
 
                \n{format_instructions}\n{query}\n
            """
            return chain_of_thought_
        else:
            return

def _chainOp(provider, model_name, temperature, strategy='zero'):

    llm = model_obj.get_llm(provider, model_name, temperature)
    
    prompt_obj = MyPrompts(strategy)
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
        elif strategy == 'cot':
            prompt += self.cot()
        else:
            raise("No prompting strategy is provided.")
        return prompt
    
    def zero_shot(self):
        prompt = """
        You are a software engineer specializing in static analysis and bug detection in AI frameworks. Your task is to classify the following code hunk as buggy or not. Based on your analysis, decide if the code hunk is buggy or not. Please generate YES or NO response without any further explanation. <CODE HUNK START FROM HERE:>
        """
        return prompt
        
    def few_shot(self, shots):
        prompt = """
        You are a software engineer specializing in static analysis and bug detection AI frameworks. Your task is to classify the following code hunk as buggy or not. Look for any missing, improper, or insufficient checker statements within the code hunk. Checker statements are generally used to perform error handling, input validation, boundary checking, or other code safety checking. Based on your analysis, decide if the code hunk has a checker bug or not. 

        Here are some examples:
        """
        for i, shot in enumerate(shots, start=1):
            deleted_lines = shot["Deleted lines"]
            added_lines = shot["Added lines"]
            label = shot["Label"]
            commit_message = shot["Commit message"]
                
        prompt += f"""
        Example {i}:
        - Deleted lines: "{deleted_lines}"
        + Added lines: "{added_lines}"
        Label: {label}
        Commit message: "{commit_message}"
        """

        prompt += """
        Please generate YES or NO response without any further explanation.
        
        <CODE HUNK START FROM HERE:>
        
        
        """

        return prompt
        
    def cot(self):
        prompt = """
            You are a software engineer specializing in static analysis and bug detection AI frameworks. 
            Your task is to classify the following code hunk as buggy or not. 
            Follow the steps below to reason through the problem and arrive at a conclusion.
                
            1. Review the Code snippet: Examine the deleted and added lines of code to identify the modifications made.
                
            2. Identify Potential Issues: Look for any missing, improper, or insufficient checker statements within the code hunk. Checker statements are generally used to perform error handling, input validation, boundary checking, or other code safety checking..
                
            3. Analyze the Impact: Consider the impact of the identified issues on the functionality and reliability of AI frameworks. 

            4. Make a Decision: Based on the above analysis, decide if the code hunk has a checker bug or not.

            5. Please generate YES or NO response without any further explanation.
                
            <CODE HUNK START FROM HERE:> 
        """
        return prompt

def main(args):
    lib_name = args[0]
    num_iter = args[1]
    
    data_path = f"data/taxonomy_data/{lib_name}_test_data.json"
    rule_path = f"data/rule_set.json"
        
    rule_data = load_json(rule_path)
    data = load_json(data_path)
    
    prompt_obj = ConstructPrompt()
    
    for iteration in range(1, num_iter):
        
        if not os.path.exists(f'logs/{libname}/{iteration}'):
            os.makedirs(f'logs/{libname}/{iteration}')

        for provider in ['openai']:
            current_models = MODEL_NAMES[provider]
            for model_name in current_models:
                for temperature in range(11):
                    temperature = temperature/10
                    for strategy in ['zero', 'few', 'cot']:
                        
                        hisotry_file = f'logs/{libname}/{iteration}/{model_name}::{temperature}::{strategy}.txt'
                        f1 = open(hisotry_file, 'a')
                        hist = read_txt(f'logs/{libname}/{iteration}/{model_name}::{temperature}::{strategy}.txt')
                        
                        for idx, instance in enumerate(data):
                            if instance['commit_link'] not in hist:
                                
                                write_list_to_txt(instance['commit_link'], f'logs/{libname}/{iteration}/{model_name}::{temperature}::{strategy}.txt')

                                if strategy == 'few':
                                    rand_num = random.randint(1, 13)
                                    _shot = [rule_data[f"entry{rand_num}"]['example1'], rule_data[f"entry{rand_num}"]['example2']]
                                    if instance['commit_link'] == _shot[0]['commit_link'] or instance['commit_link'] == _shot[1]['commit_link']:
                                        print(Fore.RED + "This instance is among one of the shots, so I am skipping this one!")
                                        continue
                                        
                                    prompt_template = prompt_obj.get(strategy, _shot)
                                else:
                                    prompt_template = prompt_obj.get(strategy)                                           
                                            
                                            
                                for change in instance['changes']:
                                    if not change:
                                            continue
                                        
                                    if 'test' in change['path'] or 'tests' in change['path']:
                                            continue
                                        
                                    for k, patch in enumerate(change['patches']):
                                        if not patch:
                                            continue
                                            
                                        llm = model_obj.get_llm(provider, model_name, temperature)
                                            
                                        output = llm.invoke(f"{prompt_template} {patch['hunk']}")            
                                        
                                        try:    
                                            decision = is_buggy(output.content)
                                        except Exception as e:
                                            print(Fore.RED + e)
                                        
                                        output = [iteration, lib_name, model_name, temperature, strategy, instance['commit_link'], change['path'], True, decision]
                                                    
                                        write_to_csv(output, lib_name)
                                            
                                        print(Fore.CYAN + f"Iteration: {iteration} | Provider: {provider} | Model: {model_name} | Temp: {temperature} | Strategy: {strategy} | Instance: {idx}/{len(data)} | Patch Number: {k+1} | Decision: {decision}" + Style.RESET_ALL)
                                        time.sleep(2)

                                
if __name__ == '__main__':
    libname = sys.argv[1]
    num_iter = sys.argv[2]
    # libname = 'pytorch'
    # num_iter = 5
    args = [libname, int(num_iter)]
    main(args)
