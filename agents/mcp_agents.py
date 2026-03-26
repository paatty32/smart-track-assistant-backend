import logging

from dotenv import load_dotenv
from llama_index.core.agent import FunctionAgent
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI


logging.basicConfig(format='%(asctime)s - %(module)s - %(funcName)s: %(message)s', level=logging.INFO)
#Eigener Logger für das Python Modul. __name__ = name des Python Moduls
logger = logging.getLogger(__name__)

system_prompt= ("""
Systemprompt für den Sprint-Trainingsplan-Agenten

Zweck: Erstelle Sprint-Trainingspläne unter Berücksichtigung von Wetterdaten und Informationen aus einer Vektordatenbank.

Zielgruppe: B-Trainer im Sprint.

Technische Details:

Nutze Wetterdaten vom MCP-Server.
Verwende eingebettete Informationen aus einer PDF im Vektorstore.
Nutze das Tool create_training_plan, um Daten aus dem Kontext zu extrahieren.
Nutze das Tool insertTrainingPlan, um den Trainingsplan nur bei expliziter Aufforderung durch den Benutzer in die Datenbank zu speichern.
Antwortformat:

Datum
Wetter
Einheit:
Aufwärmen: [Stichpunkte]
Hauptteil: [Stichpunkte]

Analysiere Wetterdaten für den angegebenen Tag.
Identifiziere Trainingsschwerpunkte und plane entsprechende Einheiten.
Passe Pläne an extreme Wetterbedingungen an.
Verwende create_training_plan, um den aktuellen Trainingsplan aus dem Kontext zu extrahieren.
Speichere den Trainingsplan mit insertTrainingPlan nur bei expliziter Aufforderung durch den Benutzer in der Datenbank.
Ton und Stil: Sachlich und professionell.
""")

def create_ollama_agent(tools, vector_tools):
    logger.info("START - Ollama Agent erstellen.")
    ollama_llm = Ollama(
        model="gpt-oss:20b",
        request_timeout=120.0,
        # Manually set the context window to limit memory usage
        context_window=8000,
    )

    ollama_agent = FunctionAgent(
        llm=ollama_llm,
        tools=tools + [vector_tools],
        system_prompt=system_prompt
    )
    logger.info("ENDE - Ollama Agent erstellen.")
    logger.info("Ollama Agent wurde erstellt.")
    return ollama_agent

def create_openai_agent(tools, vector_tools):
    logger.info("START - OpenAI Agent erstellen.")
    load_dotenv()
    openai_llm = OpenAI(model="gpt-4o-mini")

    openai_agent = FunctionAgent(
        llm=openai_llm,
        tools=tools + [vector_tools],
        system_prompt=system_prompt
    )
    logger.info("ENDE - OpenAI Agent erstellen.")
    logger.info("OpenAI Agent wurde erstellt.")
    return openai_agent

def create_agent(tools, vector_tools, args, settings):
    args = args
    mode = args.agent if args.agent else settings.agent_mode
    agent = AGENT_FACTORY[mode](tools, vector_tools)
    return agent

AGENT_FACTORY = {
    "ollama": create_ollama_agent,
    "openai": create_openai_agent,
}