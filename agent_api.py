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
    
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Replace this with your LangGraph agent logic
def run_langgraph_agent(user_message: str) -> str:
    # Dummy response — replace with your real agent call
    return f"Echo from LangGraph agent: {user_message}"

# agent_manager = AgentManager()

async def get_response(input):
    # Initialize the agent manager
    # await agent_manager.connect_to_server()
        
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
        state = await get_response(payload['messages'][-1]['content'])
        # Extract the last message from the agent's response
        content = state['messages'][-1].content if len(state['messages']) > 0 else ""
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
        }

    # --- streaming path ------------------------------------------------------
    async def event_stream():
        state = await get_response(payload['messages'][-1]['content'])
        # Extract the response content from the AIMessage in full_text
        # Use the last message in the list for more reliability
        response_content = state['messages'][-1].content if len(state['messages']) > 0 else ""
        print("Agent response:", response_content)
        
        # send first chunk (delta) - role only
        chunk = {
            "id": f"chatcmpl-agent-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "langgraph-agent",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # Smart tokenization - handle markdown and code blocks specially
        tokens = smart_tokenize(response_content)
        
        # Stream tokens with natural typing speed
        for token in tokens:
            chunk["choices"][0]["delta"] = {"content": token}
            yield f"data: {json.dumps(chunk)}\n\n"
            
            # Variable typing speed based on token characteristics
            delay = calculate_typing_delay(token)
            await asyncio.sleep(delay)
        
        # final chunk with finish_reason
        chunk["choices"][0].update({"delta": {}, "finish_reason": "stop"})
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Optional: root health check
@app.get("/")
def health():
    return {"status": "running"}

# Helper functions for improved streaming

def smart_tokenize(text: str) -> List[str]:
    """
    Intelligently tokenize text to preserve markdown formatting and natural language flow.
    """
    # Handle code blocks specially
    code_block_pattern = r'(```[\s\S]*?```)'
    parts = re.split(code_block_pattern, text)
    
    result = []
    for part in parts:
        if part.startswith('```') and part.endswith('```'):
            # Keep code blocks mostly intact but split into lines for better UX
            lines = part.split('\n')
            for i, line in enumerate(lines):
                if i == 0 or i == len(lines) - 1:  # First line (```language) or last line (```)
                    result.append(line + '\n')
                else:
                    # Process each code line
                    result.append(line + '\n')
        else:
            # Handle markdown headers specially
            header_pattern = r'(^#{1,6}\s.*$)'
            md_parts = re.split(header_pattern, part, flags=re.MULTILINE)
            
            for md_part in md_parts:
                if re.match(header_pattern, md_part, re.MULTILINE):
                    # Keep headers intact
                    result.append(md_part)
                else:
                    # Split normal text into sentences first, then into smaller chunks
                    sentences = re.split(r'([.!?]\s)', md_part)
                    for i in range(0, len(sentences), 2):
                        sentence = sentences[i]
                        ending = sentences[i + 1] if i + 1 < len(sentences) else ''
                        
                        # Now split the sentence into meaningful chunks
                        # Keep punctuation with words, split on spaces
                        chunks = re.findall(r'\S+\s*', sentence + ending)
                        result.extend(chunks)
    
    return result

def calculate_typing_delay(token: str) -> float:
    """
    Calculate a natural typing delay based on token characteristics:
    - Longer tokens get slightly longer delays
    - Special markdown tokens get shorter delays (makes formatting appear faster)
    - Random variation added for natural feel
    """
    # Base delay
    base_delay = 0.03
    
    # Adjust for token length (longer tokens = slightly longer delay)
    length_factor = min(len(token) / 10, 1.0)
    
    # Shorter delays for markdown formatting and symbols
    is_markdown = bool(re.match(r'^[#*_`~\[\]()]+$', token.strip()))
    markdown_factor = 0.5 if is_markdown else 1.0
    
    # Punctuation gets shorter delays
    is_punctuation = bool(re.match(r'^[.,;:!?-]+$', token.strip()))
    punctuation_factor = 0.3 if is_punctuation else 1.0
    
    # New line gets a slightly longer pause
    newline_factor = 1.5 if '\n' in token else 1.0
    
    # Add randomness for natural feel (±20%)
    random_factor = random.uniform(0.8, 1.2)
    
    delay = base_delay * length_factor * markdown_factor * punctuation_factor * newline_factor * random_factor
    
    # Ensure minimum delay
    return max(delay, 0.01)

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)

