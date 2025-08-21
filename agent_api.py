from contextlib import AsyncExitStack
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Generator, AsyncGenerator, Optional
import uvicorn
import time
import asyncio
import json
import random
import re
from backend.src.agent.graph_v1 import graph
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
# Add the langgraph config import for custom streaming
from langgraph.config import get_stream_writer

load_dotenv()

# Add the custom stream generation function following langgraph example
def generate_custom_stream(type: Literal["think","normal"], content: str):
    """Generate custom stream content for think or normal responses"""
    content = "\n"+content+"\n"
    custom_stream_writer = get_stream_writer()
    return custom_stream_writer({type: content})

class State(TypedDict):
    messages: Annotated[list, add_messages]

async def get_response(input):
    # Process message with agent using ainvoke for proper async handling
    # This prevents blocking the event loop
    state = await graph.ainvoke({
        "messages": [{"role": "user", "content": input}], 
        "max_research_loops": 5, 
        "initial_search_query_count": 5
    })
    print("Agent state received")
    return state

# FastAPI app
app = FastAPI(
    title="Langgraph API",
    description="Langgraph API with custom streaming support",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to ["http://localhost:3000"] for safety
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add test endpoint following langgraph example
@app.get("/test")
async def test():
    return {"message": "Hello World"}

# Add the new /stream endpoint following langgraph example pattern
@app.post("/stream")
async def stream(inputs: State):
    """Stream endpoint following langgraph example structure"""
    async def event_stream():
        try:
            stream_start_msg = {
                'choices': 
                    [
                        {
                            'delta': {}, 
                            'finish_reason': None
                        }
                    ]
                }

            # Stream start
            yield f"data: {json.dumps(stream_start_msg)}\n\n"            

            # Processing langgraph stream response with <think> block support
            async for event in graph.astream(input=inputs, stream_mode="custom"):
                print(event)
                think_content = event.get("think", None)
                normal_content = event.get("normal", None)
    
                think_msg = {
                    'choices': 
                    [
                        {
                            'delta':
                            {
                                'reasoning_content': think_content, 
                            },
                            'finish_reason': None                            
                        }
                    ]
                }

                normal_msg = {
                    'choices': 
                    [
                        {
                            'delta':
                            {
                                'content': normal_content, 
                            },
                            'finish_reason': None                            
                        }
                    ]
                }

                yield f"data: {json.dumps(think_msg)}\n\n"
                yield f"data: {json.dumps(normal_msg)}\n\n"

            # End of the stream
            stream_end_msg = {
                'choices': [ 
                    {
                        'delta': {}, 
                        'finish_reason': 'stop'
                    }
                ]
            }
            yield f"data: {json.dumps(stream_end_msg)}\n\n"

        except Exception as e:
            # Simply print the error information
            print(f"An error occurred: {e}")

    return StreamingResponse(
        event_stream(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# Optional: root health check
@app.get("/")
def health():
    return {"status": "running"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)

