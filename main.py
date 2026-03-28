import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from llama_index.core.agent import AgentStream
from starlette.websockets import WebSocket, WebSocketDisconnect
from llama_index.core.workflow import Context

from agents.mcp_agents import create_agent
from database.databaseConnector import get_enginge, create_tables
from utils.cli import parse_args
from utils.deps import get_agent, get_context
from mcp_utils.mcp_client import get_smart_track_assistant_tools
from utils.settings import get_settings
from vectorStoreIndex.vectorStoreIndex import build_query_engine, build_index_tool

import uvicorn

logging.basicConfig(format='%(asctime)s - %(module)s - %(funcName)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starte MCP Server...")
    load_dotenv()

    settings = get_settings()

    #CLI holen.
    args = parse_args()

    #Tools vom MCP laden
    tools = await get_smart_track_assistant_tools()
    logger.info(f"Tools: {tools}")

    logger.info("Initialisiere Vector Index...")
    query_engine = build_query_engine(args)
    index_tool = build_index_tool(query_engine)

    logger.info("Initialisiere Agent..")
    #Agent initialisieren
    agent = create_agent(tools, index_tool, args, settings)
    logger.info("Agent fertig initialisiert")
    #Context erstellen
    agent_ctx = Context(agent)

    #Datenbank connection
    logger.info("Datenbank Connection")
    #TODO: Auslagern in den MCP server
    enginge = get_enginge()
    create_tables(enginge)

    logger.info("Datenbank Connection fertig")

    ##state globaler Speicher
    app.state.agent = agent
    app.state.context = agent_ctx

    # App läuft
    yield

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket,
                             ollamaAgent=Depends(get_agent),
                             context=Depends(get_context)):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received: {data}")
            response = ollamaAgent.run(user_msg=data, ctx=context)

            #Agent-Schicht:
            async for r in response.stream_events():
                if isinstance(r, AgentStream):
                    await websocket.send_text(str(r.delta))

            await websocket.send_text("[DONE]")

            state = await context.store.get("")
            if state and "plan" in state:
                pending_plan = state["plan"]
                logger.info(f"Neuer Trainingsplan im Context: {pending_plan}")

    except WebSocketDisconnect:
        logger.info("Disconnected")

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8001, reload=True)