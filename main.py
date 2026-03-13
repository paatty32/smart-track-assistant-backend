import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI
from llama_index.core.agent import AgentStream
from starlette.websockets import WebSocket, WebSocketDisconnect
from llama_index.core.workflow import Context

from agents.OllamaAgent import createOllamaAgent
from database.databaseConnector import get_enginge, create_tables
from mcp_utils.mcp_client import get_weather_tools
from vectorStoreIndex.vectorStoreIndex import build_query_engine, build_index_tool

mcp_process: subprocess.Popen | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starte MCP Server...")

    #Tools vom MCP laden
    tools = await get_weather_tools()
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

    print("Datenbank Connection fertig")

    ##state globaler Speicher
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