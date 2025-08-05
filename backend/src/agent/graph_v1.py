import os

from backend.src.agent.tools_and_schemas import SearchQueryList, Reflection,QueryNecessary
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client
# Add import for custom streaming
from langgraph.config import get_stream_writer
from typing import Literal

from backend.src.agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from backend.src.agent.configuration import Configuration
from backend.src.agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek

from backend.src.agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
from typing import TypedDict, List, Annotated, Literal, Dict, Union, Optional 
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
from langchain_core.runnables import RunnableLambda, RunnableSequence
import json

# Add the custom stream generation function
def generate_custom_stream(type: Literal["think","normal"], content: str):
    """Generate custom stream content for think or normal responses"""
    content = "\n"+content+"\n"
    custom_stream_writer = get_stream_writer()
    if custom_stream_writer:
        return custom_stream_writer({type: content})
    
def question_necessary(state: OverallState):
    """Node function that determines if web search is necessary"""
    llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=1,
    timeout=None,
    max_retries=2,
    )
    prompt = """
    You are a helpful assistant to decide whether the question needs search materials from the net.
    If the question needs search materials from the net, return 'yes'.
    If the question does not need search materials from the net you can answer it directly, return 'no'.
    Question: {question}
    """
    prompt = prompt.format(question=state["messages"][-1].content)
    structured_llm = llm.with_structured_output(QueryNecessary)

    result = structured_llm.invoke(prompt)
    score = result.binary_score
    print(score)
    
    # Stream thinking content about the decision
    generate_custom_stream("think", f"Analyzing question: '{state['messages'][-1].content}'. Determining if web search is necessary...")
    
    if score == 'yes':
        generate_custom_stream("think", "Question requires web search. Proceeding to research phase.")
        return {"needs_search": True}
    else:
        generate_custom_stream("think", "Question can be answered directly without web search.")
        return {"needs_search": False}

def route_question(state: OverallState):
    """Conditional edge function that routes based on question_necessary result"""
    needs_search = state.get("needs_search", False)
    print('needs_search',needs_search)
    if needs_search:
        return "generate_query"
    else:
        return "easy_answer"


def generate_query(state: OverallState, config: RunnableConfig)->QueryGenerationState :
    """LangGraph node that generates a search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search query for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # Stream thinking content about query generation
    research_topic = get_research_topic(state["messages"])
    generate_custom_stream("think", f"Starting research on: '{research_topic}'. Generating {state['initial_search_query_count']} optimized search queries...")

    # init Gemini 2.0 Flash
    # llm = ChatGoogleGenerativeAI(
    #     model=configurable.query_generator_model,
    #     temperature=1.0,
    #     max_retries=2,
    #     api_key=os.getenv("GEMINI_API_KEY"),
    # )
    llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=1,
    timeout=None,
    max_retries=2,
    )
    structured_llm = llm.with_structured_output(SearchQueryList)
    print(get_research_topic(state["messages"]),)
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=research_topic,
        number_queries=state["initial_search_query_count"],
    )
    
    generate_custom_stream("think", "Creating research plan and search queries using DeepSeek...")
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    # state['plan'] = result.plan
    print(result)
    
    # Stream information about generated queries
    query_list_str = "\n".join([f"- {query}" for query in result.query])
    generate_custom_stream("think", f"Generated search queries:\n{query_list_str}")
    
    return {"plan": result.plan, "query_list": result.query}

def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]

class Citation(BaseModel):
    label: str = Field(
        ...,
        description="The name of the website.",
    )
    source_id: str = Field(
        ...,
        description="The url of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and return all the sources used."""
    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources. Include any relevant sources in the answer as markdown hyperlinks. For example: 'This is a sample text ([url website](url))'"
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )



def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configura"tion for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Stream thinking content about web research
    generate_custom_stream("think", f"Executing web search for: '{state['search_query']}'...")
    
    # Configure
    load_dotenv()
    llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=1,
    timeout=None,
    max_retries=2,
    )


    # Initialize Tavily Search Tool
    tavily_search_tool = TavilySearch(
        max_results=5,
        topic="general",
    )
    tool_calling_llm = llm.bind_tools([tavily_search_tool])
    # agent = create_react_agent(llm, [tavily_search_tool])

    # user_input = "What nation hosted the Euro 2024?"
    user_input = state["search_query"]
    # formatted_prompt = web_searcher_instructions.format(
    #     current_date=get_current_date(),
    #     research_topic='price of 5090',
    # )

    web_searcher_instructions = """Use your search tool to conduct targeted Searches to gather the most recent, credible information on "{research_topic}" and synthesize it into a verifiable text artifact.

    Instructions:
    - Query should ensure that the most current information is gathered. The current date is {current_date}.
    - Conduct multiple, diverse searches to gather comprehensive information.
    - Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
    - The output should be a well-written summary or report based on your search findings. 
    - Only include the information found in the search results, don't make up any information.

    Research Topic:
    {research_topic}
    """

    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=user_input,
    ) + "\n\nIMPORTANT: Use the search tool provided to find the most up-to-date information, and choose the right arguments for the search tool."


    def run_tool(response_msg):
        if "tool_calls" in response_msg.additional_kwargs:
            for tool_call in response_msg.additional_kwargs["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                parsed_args = json.loads(arguments)
                if tool_name == "tavily_search":
                    return {"tool_result": tavily_search_tool.invoke(parsed_args)}
        return {"tool_result": "No tool call made."}

    # === Final LLM with structured output ===
    final_llm = llm.with_structured_output(QuotedAnswer)
    # Step 3: Combine it with the LLM output step
    chain = (
        tool_calling_llm |
        RunnableLambda(run_tool) |
        (lambda inputs: final_llm.invoke(f"Based on this search result: {inputs['tool_result']}, answer the user question."))
    )

    generate_custom_stream("think", "Processing search results and extracting citations...")
    result = chain.invoke(formatted_prompt)
    web_research_result = result.answer + "\n---\n\n".join([citation.source_id for citation in result.citations])
    citations_list = []
    for citation in result.citations:
        citations_dict = {}
        citations_dict['label'] = citation.label
        citations_dict['short_url'] = citation.source_id
        citations_dict['value'] = citation.quote
        citations_list.append(citations_dict)

    # Stream completion of web research
    generate_custom_stream("think", f"Web search completed. Found {len(citations_list)} relevant sources for query: '{state['search_query']}'")
    
    return {
        "sources_gathered": citations_list,
        "search_query": [state["search_query"]],
        "web_research_result": [web_research_result],
    }

def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    # configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    # reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    # Stream thinking content about the reflection process
    generate_custom_stream("think", f"Research loop {state['research_loop_count']}: Reflecting on gathered information about '{get_research_topic(state['messages'])}'. Analyzing {len(state['web_research_result'])} research results to identify knowledge gaps...")

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
        plan=state["plan"],
    )
    # init Reasoning Model
    # llm = ChatGoogleGenerativeAI(
    #     model=reasoning_model,
    #     temperature=1.0,
    #     max_retries=2,
    #     api_key=os.getenv("GEMINI_API_KEY"),
    # )
    llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=1,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    )
    
    generate_custom_stream("think", "Evaluating research completeness and identifying follow-up queries...")
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    # Stream information about the reflection result
    if result.is_sufficient:
        generate_custom_stream("think", "Research appears sufficient. Ready to finalize answer.")
    else:
        generate_custom_stream("think", f"Knowledge gap identified: {result.knowledge_gap}. Generating {len(result.follow_up_queries)} follow-up queries for deeper research.")

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]

def easy_answer(state: OverallState):
    """Node function that provides direct answers without web search"""
    llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=1,
    timeout=None,
    max_retries=2,
    )
    
    # Stream thinking content
    generate_custom_stream("think", f"Providing direct answer to: '{state['messages'][-1].content}'")
    
    prompt = """    
    You are a helpful assistant to answer the user question.
    Question: {question}
    Answer:
    """
    prompt = prompt.format(question=state["messages"][-1].content)
    
    generate_custom_stream("think", "Generating direct answer using DeepSeek...")
    result = llm.invoke(prompt)
    
    # Stream the normal response content
    generate_custom_stream("normal", result.content)
    
    return {
        "messages": [AIMessage(content=result.content)],
    }

def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    # reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    # Stream thinking content about the finalization process
    generate_custom_stream("think", f"Finalizing research on: {get_research_topic(state['messages'])}. Analyzing {len(state['web_research_result'])} research results and {len(state['sources_gathered'])} sources to create comprehensive answer...")

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model, default to Gemini 2.5 Flash
    # llm = ChatGoogleGenerativeAI(
    #     model=reasoning_model,
    #     temperature=0,
    #     max_retries=2,
    #     api_key=os.getenv("GEMINI_API_KEY"),
    # )
    llm = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=1,
    timeout=None,
    max_retries=2,
    )
    
    generate_custom_stream("think", "Generating final comprehensive answer using DeepSeek reasoning model...")
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    # Stream the normal response content
    generate_custom_stream("normal", result.content)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("question_necessary", question_necessary)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)
builder.add_node("easy_answer", easy_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "question_necessary")



# Edges taken after the `action` node is called.
# workflow.add_conditional_edges(
#     "retrieve",
#     # Assess agent decision
#     grade_documents,
# )
# workflow.add_edge("generate_answer", END)
# workflow.add_edge("rewrite_question", "generate_query_or_respond")

builder.add_conditional_edges(
    "question_necessary",
    route_question,
    {
        "generate_query": "generate_query",
        "easy_answer": "easy_answer",
    },  
)
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)
builder.add_edge("easy_answer", END)

graph = builder.compile(name="pro-search-agent")