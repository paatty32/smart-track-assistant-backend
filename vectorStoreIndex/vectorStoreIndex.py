import logging
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, StorageContext
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "ressources", "documents")

PERSIST_DIR = None
embedding = None
llm = None

#loading
def getDocuments():
    documents = SimpleDirectoryReader(DOCUMENTS_DIR).load_data()
    return documents

def build_query_engine(args):
    global PERSIST_DIR
    global embedding
    global llm

    documents = getDocuments()

    match args.agent:
        case "openai":
            folder_name = args.agent
            embedding = OpenAIEmbedding(model="text-embedding-3-small")

            llm = OpenAI(model="gpt-4o-mini")
        case _:
            folder_name = "ollama"
            embedding = OllamaEmbedding(model_name="embeddinggemma")
            llm = Ollama(model="gpt-oss:20b",
                         request_timeout=120.0,
                         # Manually set the context window to limit memory
                         usagecontext_window=8000)

    PERSIST_DIR = os.path.join(BASE_DIR, "ressources", "index", folder_name)
    os.makedirs(PERSIST_DIR, exist_ok=True)

    # rebuild storage context
    logger.info("START - Loading Index")
    try:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        logger.info(f"{PERSIST_DIR} directory")
        # load index
        index = load_index_from_storage(storage_context, embed_model=embedding)
    except Exception as e:
        import traceback
        logger.error(f"Fehler beim laden des Index: {e}")
        logger.error(traceback.format_exc())
        index = None

    if index is None:
        logger.info("END - Loading Index")
        logger.info("START - Index is empty. Starting creating index")
        index = VectorStoreIndex.from_documents(documents, embed_model=embedding, show_progress_bar=True)
        logger.info("END - Index is empty. Creating index")
        # storing index
        logger.info("START - Storing index")
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        logger.info("END - Storing index")


    return index.as_query_engine(llm=llm)

def build_index_tool(query_engine):
    return QueryEngineTool.from_defaults(query_engine=query_engine, description="Fachliteratur über das Sprinttraining")
