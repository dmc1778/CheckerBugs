from langchain_core.prompts import PromptTemplate
from langchain_fireworks import Fireworks
from langchain_fireworks.chat_models import ChatFireworks
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

class LLMs:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        os.environ["FIREWORKS_API_KEY"] = os.getenv('FIREWORKS_API_KEY')
        os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
        
    def get_llm(self, provider: str, model_name: str, temperature: float = 0.7, max_tokens: int = 1000):
        """
        Returns an LLM instance based on the selected provider.
        
        :param provider: The LLM provider (e.g., "fireworks", "openai").
        :param model: The model identifier.
        :param temperature: Sampling temperature.
        :param max_tokens: Maximum number of tokens to generate.
        :param top_p: Nucleus sampling parameter.
        :return: An instance of the selected LLM.
        """
        
        if provider.lower() == "fireworks":
            if model_name.lower() == "accounts/fireworks/models/deepseek-r1":
                return ChatFireworks(
                    model=model_name,
                    max_tokens=max_tokens,
                )
            else:
                return ChatFireworks(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        elif provider.lower() == "openai":
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif provider.lower() == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")