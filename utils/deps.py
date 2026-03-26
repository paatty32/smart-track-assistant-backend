from fastapi import WebSocket

def get_agent(websocket: WebSocket):
    return websocket.app.state.agent

def get_context(websocket: WebSocket):
    return websocket.app.state.context