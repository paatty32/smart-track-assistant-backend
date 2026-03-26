from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

async def get_smart_track_assistant_tools():
    mcp_smart_track_client = BasicMCPClient("http://127.0.0.1:8000/mcp")
    mcp_smart_track_tools = McpToolSpec(client=mcp_smart_track_client)

    tools = await mcp_smart_track_tools.to_tool_list_async()
    return tools