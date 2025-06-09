# from fastapi import FastAPI, Request
from contextlib import AsyncExitStack
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import time
from fastapi.responses import StreamingResponse
from fastapi import Body
import asyncio
import json
from agent.graph import graph
from dotenv import load_dotenv


load_dotenv()


# class AgentManager:
#     _instance = None
#     def __init__(self):
#         # Initialize session and client objects
#         self.session: Optional[ClientSession] = None
#         self.exit_stack = AsyncExitStack()
#         # self.agent: Optional[Agent] = None
    
#     async def connect_to_server(self):
#         if self.session:
#             return
#         # Server parameters for MCP
#         server_params = StdioServerParameters(
#             command='uv',
#             args=['run', 'mcp_server.py'],
#         )

#         stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
#         self.stdio, self.write = stdio_transport
#         self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
#         await self.session.initialize()
        
#         # List available tools
#         response = await self.session.list_tools()
#         tools = response.tools
#         print("\nConnected to server with tools:", [tool.name for tool in tools])
    
#     async def create_react_agent(self):
#         if not self.session:
#             await self.connect_to_server()
#         model = ChatDeepSeek(model="deepseek-chat")

#         tools = await load_mcp_tools(self.session)
#         agent = create_react_agent(model,tools)

#         # agent_with_history = RunnableWithMessageHistory(
#         #     agent,
#         #     get_chat_history,
#         #     input_messages_key="messages",
#         #     history_messages_key="history",
#         # )
#         # agent_response = await agent.ainvoke({'messages': '分析最近的A股大盘'}, config={"configurable": {"session_id": "default"}})
#         # print(agent_response)
#         return agent
    
    

# Replace this with your LangGraph agent logic
def run_langgraph_agent(user_message: str) -> str:
    # Dummy response — replace with your real agent call
    return f"Echo from LangGraph agent: {user_message}"

# agent_manager = AgentManager()

async def get_response(input):
    # Initialize the agent manager
    # await agent_manager.connect_to_server()
        
    # Process message with agent
    # agent = await agent_manager.create_react_agent()
    # agent_response = await agent.ainvoke({'messages': '分析最近的A股大盘'})
    # state = graph.invoke({"messages": [{"role": "user", "content": "Who won the euro 2024"}], "max_research_loops": 3, "initial_search_query_count": 3})
    agent_response =  graph.invoke({'messages': [HumanMessage(content=input)]})
    print(agent_response)
    return agent_response

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to ["http://localhost:3000"] for safety
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy model listing
@app.get("/v1/models")
def get_models():
    return {
        "data": [
            {
                "id": "langgraph-agent",
                "object": "model",
                "created": 0,
                "owned_by": "you"
            }
        ],
        "object": "list"
    }

# Request/Response schema for chat/completions
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]

@app.post("/v1/chat/completions")
async def chat_completions(payload: dict = Body(...)):
    stream = payload.get("stream", False)
    # --- non-streaming path --------------------------------------------------
    if not stream:
        content = await get_response(payload['messages'][-1]['content'])
        # content = f"Echo: {payload['messages'][-1]['content']}"
        return {
            "id": "chatcmpl-agent123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "langgraph-agent",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            # "usage": {"prompt_tokens": 0, "completion_tokens": len(content.split()), "total_tokens": len(content.split())}

        }

    # --- streaming path ------------------------------------------------------
    async def event_stream():
        # full_text = f"Echo: {payload['messages'][-1]['content']}"
        full_text = await get_response(payload['messages'][-1]['content'])
        # Extract the response content from the AIMessage in full_text
        response_content = full_text['messages'][1].content if len(full_text['messages']) > 1 else ""
        print(response_content)
        # send first chunk (delta)
        full_text = response_content
        chunk = {
            "id": "chatcmpl-agent123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "langgraph-agent",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # send token-by-token (here we just split on spaces)
        for token in full_text.split():
            chunk["choices"][0]["delta"] = {"content": token + " "}
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.04)           # simulate typing

        # final chunk with finish_reason
        chunk["choices"][0].update({"delta": {}, "finish_reason": "stop"})
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Optional: root health check
@app.get("/")
def health():
    return {"status": "running"}

# Run the server
if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=9000)

