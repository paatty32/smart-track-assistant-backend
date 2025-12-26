import logging

from llama_index.core.agent import FunctionAgent, ReActAgent
from llama_index.llms.ollama import Ollama

logging.basicConfig(format='%(asctime)s - %(module)s - %(funcName)s: %(message)s', level=logging.INFO)
#Eigener Logger für das Python Modul. __name__ = name des Python Moduls
logger = logging.getLogger(__name__)


def createOllamaAgent(tools, vector_tools):
    ollama_llm = Ollama(
        model="gpt-oss:20b",
        request_timeout=120.0,
        # Manually set the context window to limit memory usage
        context_window=8000,
    )

    ollama_agent = FunctionAgent(
        llm=ollama_llm,
        tools=tools + [vector_tools]
    )

    return ollama_agent
