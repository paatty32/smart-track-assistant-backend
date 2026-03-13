import logging

from llama_index.core.agent import FunctionAgent, ReActAgent
from llama_index.llms.ollama import Ollama

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


def createOllamaAgent(tools, vector_tools):
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

    return ollama_agent
