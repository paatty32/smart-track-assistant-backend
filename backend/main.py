import asyncio
import subprocess
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI
from llama_index.core.agent import AgentStream
from sqlalchemy import text
from starlette.websockets import WebSocket, WebSocketDisconnect
from llama_index.core.workflow import Context

from agents.OllamaAgent import createOllamaAgent
from database.databaseConnector import get_enginge, create_tables
from domain.TrainingPlanCreate import TrainingPlanCreate
from mcp_utils.mcp_client import get_weather_tools
from mcp_utils.open_meteo_mcp_server import async_session
from vectorStoreIndex.vectorStoreIndex import build_query_engine, build_index_tool

mcp_process: subprocess.Popen | None = None

async def create_training_plan(
    context: Context,
    datum: date,
    wetter: str,
    aufwaermen: str,
    hauptteil: str,
) -> str:
    plan = TrainingPlanCreate(
        datum=datum,
        wetter=wetter,
        aufwaermen=aufwaermen,
        hauptteil=hauptteil,
    )

    async with context.store.edit_state() as state:
        state["plan"] = plan
        state["plan_saved"]= False

    return f"Plan wurde erstellt: {plan}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starte MCP Server...")

    #Eigener Subprocess wird gestartet TODO: Verstehen was hier passiert
    mcp_process = subprocess.Popen(
        ["python3", "-m", "mcp_utils.open_meteo_mcp_server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    await asyncio.sleep(3)

    #Tools vom MCP laden
    tools = await get_weather_tools()
    tools = tools + [create_training_plan]
    print("Tools: ", tools)

    print("Initialisiere Vector Index...")
    query_engine = build_query_engine()
    index_tool = build_index_tool(query_engine)

    print("Initialisiere Agent..")
    #Agent initialisieren
    agent = createOllamaAgent(tools, index_tool)
    print("Agent fertig initialisiert")

    #Datenbank connection
    print("Datenbank Connection")
    enginge = get_enginge()
    create_tables(enginge)

    with enginge.connect() as conn:
        result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public';"))
        print([row[0] for row in result])

    print("Datenbank Connection fertig")

    ##state globaler Speicher
    app.state.mcp_process = mcp_process
    app.state.agent = agent

    # App läuft
    yield

    print("Beende MCP Server...")
    if mcp_process.poll() is None:
        mcp_process.terminate()
        mcp_process.wait()
    print("MCP Server beendet")

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    ollamaAgent = websocket.app.state.agent

    # Persistente Session pro WebSocket
    if not hasattr(websocket, "db_session"):
        websocket.db_session = async_session()

    session = websocket.db_session

    if not hasattr(websocket, "context"):
        websocket.context = Context(ollamaAgent)

    context = websocket.context

    try:
        while True:
            data = await websocket.receive_text()
            print("Received: ", data)
            response = ollamaAgent.run(user_msg=data, ctx=context)

            #Agent-Schicht:
            async for r in response.stream_events():
                if isinstance(r, AgentStream):
                    await websocket.send_text(str(r.delta))

            await websocket.send_text("[DONE]")

            state = await context.store.get("")
            if state and "plan" in state:
                pending_plan = state["plan"]
                print("Neuer Trainingsplan im Context", pending_plan)

    except WebSocketDisconnect:
        print("Disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8001, reload=True)